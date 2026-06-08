//! Zaino-State ChainIndex unit tests.

pub(crate) mod finalised_state;
pub(crate) mod mempool;
mod mockchain_tests;
mod non_finalised_state;
mod poll;
mod proptest_blockgen;
mod sync_loop;
pub(crate) mod types;
pub(crate) mod vectors;

pub(crate) fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_timer(tracing_subscriber::fmt::time::UtcTime::rfc_3339())
        .with_target(true)
        .try_init()
        .ok();
}

use std::path::{Path, PathBuf};
use tempfile::TempDir;
use tokio::sync::OnceCell;
use tokio::time::Duration;
use zaino_common::{
    network::ActivationHeights, DatabaseConfig, DatabaseSize, Network, StorageConfig,
};
use zaino_proto::proto::utils::PoolTypeFilter;
use zebra_state::HashOrHeight;

use crate::{
    chain_index::{
        finalised_state::ZainoDB,
        finalized_height_floor,
        source::mockchain_source::MockchainSource,
        tests::poll::poll_until,
        tests::vectors::{
            build_active_mockchain_source, build_mockchain_source, copy_dir_recursive,
            load_test_vectors, sync_db_with_blockdata,
        },
        types::{BestChainLocation, Height, TransactionHash},
        ChainIndex, NodeBackedChainIndex, NodeBackedChainIndexSubscriber, SyncTimings,
    },
    status::{Status as _, StatusType},
    BlockCacheConfig,
};
use zebra_chain::{parameters::ConsensusBranchId, serialization::ZcashDeserializeInto};

/// Selects which factory the test setup uses to build its `MockchainSource`,
/// which in turn determines the source's `active_chain_height` and so the
/// indexer's sync target.
///
/// - `Active` → `build_active_mockchain_source(150, blocks)`: source has a
///   separately-tracked `active_height = 150` that tests can advance via
///   `mockchain.mine_blocks(N)`. Indexer's finalised sync target is
///   `finalized_height_floor(150) = 50`.
/// - `Static` → `build_mockchain_source(blocks)`: every loaded block is
///   immediately active (`active_height = tip_height = 200` for the 201-block
///   vector); the tip doesn't move during the test. Indexer's finalised sync
///   target is `finalized_height_floor(200) = 100`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MockchainMode {
    Active,
    Static,
}

async fn load_test_vectors_and_sync_chain_index(
    mode: MockchainMode,
) -> (
    Vec<vectors::TestVectorBlockData>,
    NodeBackedChainIndex<MockchainSource>,
    NodeBackedChainIndexSubscriber<MockchainSource>,
    MockchainSource,
) {
    // 25 ms setup-poll interval mirrors `_with_timings`. The previous 2 s
    // value was load-bearing for the teardown race tracked in #1098: most
    // callers (mockchain_tests, mempool, poll, proptest_blockgen) drop the
    // indexer without calling `shutdown()`, and the old worker needed to
    // be parked in its post-success interval-sleep before runtime teardown
    // raced a mid-iter LMDB write. With `Drop for NodeBackedChainIndex`
    // firing `cancel_token.cancel()` and the worker's iter body wrapped in
    // `tokio::select!` against that token, the worker now exits at its
    // next await checkpoint on drop — the harness no longer needs to
    // bait the timing.
    load_with_settings(mode, SyncTimings::default(), Duration::from_millis(25)).await
}

async fn load_test_vectors_and_sync_chain_index_with_timings(
    mode: MockchainMode,
    sync_timings: SyncTimings,
) -> (
    Vec<vectors::TestVectorBlockData>,
    NodeBackedChainIndex<MockchainSource>,
    NodeBackedChainIndexSubscriber<MockchainSource>,
    MockchainSource,
) {
    load_with_settings(mode, sync_timings, Duration::from_millis(25)).await
}

async fn load_with_settings(
    mode: MockchainMode,
    sync_timings: SyncTimings,
    setup_poll_interval: Duration,
) -> (
    Vec<vectors::TestVectorBlockData>,
    NodeBackedChainIndex<MockchainSource>,
    NodeBackedChainIndexSubscriber<MockchainSource>,
    MockchainSource,
) {
    init_tracing();

    let blocks = load_test_vectors().unwrap().blocks;

    let source = match mode {
        MockchainMode::Active => build_active_mockchain_source(150, blocks.clone()),
        MockchainMode::Static => build_mockchain_source(blocks.clone()),
    };

    let temp_dir: TempDir = tempfile::tempdir().unwrap();
    let db_path: PathBuf = temp_dir.path().to_path_buf();

    // Seed the temp DB from a process-wide pre-synced fixture. The fixture is
    // built once per mode (see `v1_finalised_seed_dir`) and synced to exactly
    // the height the indexer's sync loop would target here, so spawning the
    // indexer against this copy hits a no-op `sync_to_height` and the wait
    // loop below completes on its first probe rather than after a fresh
    // ingest of every test-vector block.
    let seed = v1_finalised_seed_dir(mode).await;
    copy_dir_recursive(seed, &db_path).unwrap();

    let config = BlockCacheConfig {
        storage: StorageConfig {
            database: DatabaseConfig {
                path: db_path,
                ..Default::default()
            },
            ..Default::default()
        },
        db_version: 1,
        network: Network::Regtest(ActivationHeights::default()),
    };

    let indexer = NodeBackedChainIndex::new_with_sync_timings(source.clone(), config, sync_timings)
        .await
        .unwrap();
    let index_reader = indexer.subscriber();

    // Wait until the indexer's non-finalised state has been built and its
    // best tip matches the source. The previous form checked only
    // `finalized_state.db_height() == finalized_height_floor(active_height)`,
    // which the seed copy makes true *before* the sync loop has had a chance
    // to initialise NFS. Tests that read the NFS immediately after the
    // helper returns (`nfs_lowest_block_matches_finalized_db_tip`,
    // `sync_blocks_after_startup`, …) then unwrap on `None`. The NFS being
    // at `source.active_height()` implies the finalised DB has reached its
    // floor — the sync loop only initialises NFS after `sync_to_height`
    // succeeds — so this condition subsumes the old one.
    let expected_nfs_tip = source.active_height();
    loop {
        let nfs_ready = match index_reader.snapshot_nonfinalized_state().await {
            Ok(snap) => snap
                .get_nfs_snapshot()
                .is_some_and(|nfs| nfs.best_tip.height.0 == expected_nfs_tip),
            Err(_) => false,
        };
        if nfs_ready {
            break;
        }
        tokio::time::sleep(setup_poll_interval).await;
    }

    (blocks, indexer, index_reader, source)
}

/// Process-wide cached, fully-synced v1 finalised-state databases — one per
/// `MockchainMode`. The two modes target different heights (Active → 50,
/// Static → 100), so they need distinct seeds.
///
/// Built lazily on first call via `tokio::sync::OnceCell`, which serialises
/// the build under concurrent test access. Each test still gets an isolated
/// writable DB by copying the seed dir into its own tempdir (see
/// `copy_dir_recursive`); the seed itself is never mutated after first build.
static V1_SEED_ACTIVE: OnceCell<TempDir> = OnceCell::const_new();
static V1_SEED_STATIC: OnceCell<TempDir> = OnceCell::const_new();

fn zero_database_config(db_path: PathBuf) -> BlockCacheConfig {
    BlockCacheConfig {
        storage: StorageConfig {
            database: DatabaseConfig {
                path: db_path,
                size: DatabaseSize(0),
            },
            ..Default::default()
        },
        db_version: 1,
        network: Network::Regtest(ActivationHeights::default()),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn zero_database_size_disables_finalized_db_sync() {
    init_tracing();

    let blocks = load_test_vectors().unwrap().blocks;
    let source = build_mockchain_source(blocks);
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().to_path_buf();

    let config = zero_database_config(db_path.clone());

    let indexer =
        NodeBackedChainIndex::new_with_sync_timings(source.clone(), config, SyncTimings::fast())
            .await
            .unwrap();
    let index_reader = indexer.subscriber();

    poll_until(
        "zero database indexer ready",
        Duration::from_secs(10),
        Duration::from_millis(25),
        || async { (index_reader.status() == StatusType::Ready).then_some(()) },
    )
    .await;
    assert!(
        !db_path.join("regtest").exists(),
        "zero database size should not create or sync an on-disk finalized DB"
    );

    let tip = index_reader.best_tip_from_source().await.unwrap();
    assert_eq!(tip.height.0, source.active_height());

    let compact_block = index_reader
        .get_compact_block_from_source(
            HashOrHeight::Height(zebra_chain::block::Height(source.active_height())),
            &PoolTypeFilter::includes_all(),
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(compact_block.height, u64::from(source.active_height()));

    indexer.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn combined_status_does_not_poison_parent_lifecycle_status() {
    init_tracing();

    let blocks = load_test_vectors().unwrap().blocks;
    let source = build_active_mockchain_source(150, blocks);
    let temp_dir = tempfile::tempdir().unwrap();
    let config = zero_database_config(temp_dir.path().to_path_buf());

    let indexer =
        NodeBackedChainIndex::new_with_sync_timings(source.clone(), config, SyncTimings::fast())
            .await
            .unwrap();
    let index_reader = indexer.subscriber();

    poll_until(
        "zero database indexer ready before mempool refresh",
        Duration::from_secs(10),
        Duration::from_millis(25),
        || async { (index_reader.status() == StatusType::Ready).then_some(()) },
    )
    .await;

    source.mine_blocks(1);

    poll_until(
        "mempool refresh entered syncing state",
        Duration::from_secs(10),
        Duration::from_millis(10),
        || async { (indexer.mempool.status() == StatusType::Syncing).then_some(()) },
    )
    .await;

    assert_eq!(indexer.status(), StatusType::Syncing);

    poll_until(
        "mempool refresh returned ready",
        Duration::from_secs(10),
        Duration::from_millis(25),
        || async { (indexer.mempool.status() == StatusType::Ready).then_some(()) },
    )
    .await;

    assert_eq!(indexer.status(), StatusType::Ready);
    assert_eq!(index_reader.status(), StatusType::Ready);

    indexer.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn zero_database_size_passthrough_serves_tip_transactions_and_mempool() {
    init_tracing();

    let blocks = load_test_vectors().unwrap().blocks;
    let source = build_active_mockchain_source(150, blocks.clone());
    let temp_dir = tempfile::tempdir().unwrap();
    let config = zero_database_config(temp_dir.path().to_path_buf());

    let indexer =
        NodeBackedChainIndex::new_with_sync_timings(source.clone(), config, SyncTimings::fast())
            .await
            .unwrap();
    let index_reader = indexer.subscriber();

    poll_until(
        "zero database passthrough indexer ready",
        Duration::from_secs(10),
        Duration::from_millis(25),
        || async { (index_reader.status() == StatusType::Ready).then_some(()) },
    )
    .await;

    let snapshot = index_reader.snapshot_nonfinalized_state().await.unwrap();
    assert!(
        snapshot.get_nfs_snapshot().is_none(),
        "zero database size should not require a local non-finalized snapshot"
    );

    let tip_height = source.active_height();
    let tip_block = &blocks[tip_height as usize].zebra_block;
    let tip_compact_block = index_reader
        .get_compact_block_from_source(
            HashOrHeight::Hash(tip_block.hash()),
            &PoolTypeFilter::includes_all(),
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(tip_compact_block.height, u64::from(tip_height));

    let mined_transaction = tip_block
        .transactions
        .iter()
        .find(|tx| !tx.is_coinbase())
        .unwrap_or(&tip_block.transactions[0]);
    let mined_txid = TransactionHash::from(mined_transaction.hash());
    let (mined_bytes, _mined_branch_id) = index_reader
        .get_raw_transaction(&snapshot, &mined_txid)
        .await
        .unwrap()
        .unwrap();
    let decoded_mined = mined_bytes
        .zcash_deserialize_into::<zebra_chain::transaction::Transaction>()
        .unwrap();
    assert_eq!(mined_transaction.as_ref(), &decoded_mined);

    let (mined_location, mined_nonbest_locations) = index_reader
        .get_transaction_status(&snapshot, &mined_txid)
        .await
        .unwrap();
    assert_eq!(
        mined_location,
        Some(BestChainLocation::Block(
            tip_block.hash().into(),
            Height(tip_height)
        ))
    );
    assert!(mined_nonbest_locations.is_empty());

    let mempool_height = tip_height + 1;
    let mempool_transaction = blocks[mempool_height as usize]
        .zebra_block
        .transactions
        .iter()
        .find(|tx| !tx.is_coinbase())
        .expect("test vector next block contains a mempool transaction");
    let mempool_txid = TransactionHash::from(mempool_transaction.hash());

    let (mempool_bytes, mempool_branch_id) = poll_until(
        "zero-size passthrough mempool transaction",
        Duration::from_secs(10),
        Duration::from_millis(25),
        || async {
            index_reader
                .get_raw_transaction(&snapshot, &mempool_txid)
                .await
                .ok()
                .flatten()
        },
    )
    .await;
    let decoded_mempool = mempool_bytes
        .zcash_deserialize_into::<zebra_chain::transaction::Transaction>()
        .unwrap();
    assert_eq!(mempool_transaction.as_ref(), &decoded_mempool);
    assert_eq!(
        mempool_branch_id,
        ConsensusBranchId::current(
            &Network::Regtest(ActivationHeights::default()).to_zebra_network(),
            zebra_chain::block::Height(mempool_height)
        )
        .map(u32::from)
    );

    let (mempool_location, mempool_nonbest_locations) = index_reader
        .get_transaction_status(&snapshot, &mempool_txid)
        .await
        .unwrap();
    assert_eq!(
        mempool_location,
        Some(BestChainLocation::Mempool(Height(mempool_height)))
    );
    assert!(mempool_nonbest_locations.is_empty());

    indexer.shutdown().await.unwrap();
}

async fn v1_finalised_seed_dir(mode: MockchainMode) -> &'static Path {
    let cell = match mode {
        MockchainMode::Active => &V1_SEED_ACTIVE,
        MockchainMode::Static => &V1_SEED_STATIC,
    };
    cell.get_or_init(|| async move {
        let blocks = load_test_vectors().unwrap().blocks;
        let source = match mode {
            MockchainMode::Active => build_active_mockchain_source(150, blocks.clone()),
            MockchainMode::Static => build_mockchain_source(blocks.clone()),
        };
        let target = finalized_height_floor(source.active_height()).0;

        let temp_dir: TempDir = tempfile::tempdir().unwrap();
        let config = BlockCacheConfig {
            storage: StorageConfig {
                database: DatabaseConfig {
                    path: temp_dir.path().to_path_buf(),
                    ..Default::default()
                },
                ..Default::default()
            },
            db_version: 1,
            network: Network::Regtest(ActivationHeights::default()),
        };

        let zaino_db = ZainoDB::spawn(config, source).await.unwrap();
        sync_db_with_blockdata(zaino_db.router(), blocks, Some(target)).await;
        zaino_db.wait_until_ready().await;
        zaino_db.shutdown().await.unwrap();

        temp_dir
    })
    .await
    .path()
}
