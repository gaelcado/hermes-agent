from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


class _Adapter:
    STREAM_EDIT_INTERVAL_FLOOR = 0.4


def test_stream_consumer_applies_adapter_edit_interval_floor():
    consumer = GatewayStreamConsumer(
        _Adapter(),
        "123",
        StreamConsumerConfig(edit_interval=0.1, buffer_threshold=40, cursor=" ▉"),
    )

    assert consumer.cfg.edit_interval == 0.4
