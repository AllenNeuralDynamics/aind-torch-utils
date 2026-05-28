import pytest

from aind_torch_utils.monitoring import BlockProgressMonitor


DIAGNOSTIC_TIMING_KEYS = [
    "prep_queue_put_wait_s",
    "prep_queue_residence_s",
    "gpu_transfer_overhead_s",
    "write_queue_put_wait_s",
    "write_queue_residence_s",
    "output_ready_wait_s",
]


def test_block_progress_monitor_summarizes_completed_blocks():
    monitor = BlockProgressMonitor(total_blocks=3, t0=0.0)

    for i, (prep_s, inference_s, write_s, complete_s) in enumerate(
        [
            (1.0, 2.0, 3.0, 10.0),
            (3.0, 4.0, 5.0, 20.0),
            (5.0, 6.0, 7.0, 30.0),
        ]
    ):
        block_idx = (0, 0, i)
        monitor.start_block(block_idx, now_s=0.0)
        monitor.record_preparation_time(block_idx, prep_s)
        monitor.add_inference_time(block_idx, inference_s / 2.0)
        monitor.add_inference_time(block_idx, inference_s / 2.0)
        monitor.add_write_time(block_idx, write_s)
        monitor.complete_block(block_idx, now_s=complete_s)

    data = monitor.get_data(now_s=40.0)

    assert data["total_blocks"] == 3
    assert data["completed_blocks"] == 3
    assert data["incomplete_blocks"] == 0
    assert data["percent_complete"] == pytest.approx(100.0)
    assert data["eta_s"] == pytest.approx(0.0)

    prep_summary = data["timing_summary"]["preparation_s"]
    assert prep_summary["count"] == 3
    assert prep_summary["min"] == pytest.approx(1.0)
    assert prep_summary["mean"] == pytest.approx(3.0)
    assert prep_summary["max"] == pytest.approx(5.0)
    assert prep_summary["p50"] == pytest.approx(3.0)
    assert prep_summary["p95"] == pytest.approx(4.8)

    inference_summary = data["timing_summary"]["inference_s"]
    assert inference_summary["count"] == 3
    assert inference_summary["mean"] == pytest.approx(4.0)

    total_summary = data["timing_summary"]["total_block_processing_s"]
    assert total_summary["count"] == 3
    assert total_summary["min"] == pytest.approx(10.0)
    assert total_summary["max"] == pytest.approx(30.0)

    for key in DIAGNOSTIC_TIMING_KEYS:
        assert key in data["timing_summary"]
        assert data["timing_summary"][key]["count"] == 0


def test_block_progress_monitor_accumulates_pipeline_diagnostics():
    monitor = BlockProgressMonitor(total_blocks=1, t0=0.0)
    block_idx = (0, 0, 0)

    monitor.start_block(block_idx, now_s=0.0)
    monitor.add_prep_queue_put_wait_time(block_idx, 0.1)
    monitor.add_prep_queue_put_wait_time(block_idx, 0.2)
    monitor.add_prep_queue_residence_time(block_idx, 1.0)
    monitor.add_prep_queue_residence_time(block_idx, 2.0)
    monitor.add_gpu_transfer_overhead_time(block_idx, 0.4)
    monitor.add_gpu_transfer_overhead_time(block_idx, 0.6)
    monitor.add_write_queue_put_wait_time(block_idx, 0.05)
    monitor.add_write_queue_put_wait_time(block_idx, 0.15)
    monitor.add_write_queue_residence_time(block_idx, 2.0)
    monitor.add_write_queue_residence_time(block_idx, 3.0)
    monitor.add_output_ready_wait_time(block_idx, 0.25)
    monitor.add_output_ready_wait_time(block_idx, 0.75)
    monitor.complete_block(block_idx, now_s=10.0)

    timing_summary = monitor.get_data(now_s=10.0)["timing_summary"]

    expected_means = {
        "prep_queue_put_wait_s": 0.3,
        "prep_queue_residence_s": 3.0,
        "gpu_transfer_overhead_s": 1.0,
        "write_queue_put_wait_s": 0.2,
        "write_queue_residence_s": 5.0,
        "output_ready_wait_s": 1.0,
    }
    for key, expected_mean in expected_means.items():
        assert timing_summary[key]["count"] == 1
        assert timing_summary[key]["mean"] == pytest.approx(expected_mean)


def test_block_progress_monitor_reports_incomplete_blocks_and_eta():
    monitor = BlockProgressMonitor(total_blocks=2, t0=0.0)

    monitor.start_block((0, 0, 0), now_s=0.0)
    monitor.record_preparation_time((0, 0, 0), 1.0)
    monitor.add_inference_time((0, 0, 0), 2.0)
    monitor.add_write_time((0, 0, 0), 3.0)
    monitor.complete_block((0, 0, 0), now_s=5.0)

    monitor.start_block((0, 0, 1), now_s=6.0)
    monitor.record_preparation_time((0, 0, 1), 1.5)

    data = monitor.get_data(now_s=10.0)

    assert data["completed_blocks"] == 1
    assert data["incomplete_blocks"] == 1
    assert data["active_blocks"] == 1
    assert data["percent_complete"] == pytest.approx(50.0)
    assert data["blocks_per_sec"] == pytest.approx(0.1)
    assert data["eta_s"] == pytest.approx(10.0)
    assert data["timing_summary"]["total_block_processing_s"]["count"] == 1


def test_block_progress_monitor_reports_unknown_eta_before_completion():
    monitor = BlockProgressMonitor(total_blocks=1, t0=0.0)

    data = monitor.get_data(now_s=10.0)

    assert data["completed_blocks"] == 0
    assert data["incomplete_blocks"] == 1
    assert data["blocks_per_sec"] == pytest.approx(0.0)
    assert data["eta_s"] is None
    assert "ETA unknown" in monitor.format_progress(now_s=10.0)
