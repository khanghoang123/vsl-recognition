from vsl_recognition.audit import source_group


def test_source_group_collapses_filename_variants():
    assert source_group({"video_name": "12345.mp4"}) == "12345"
    assert source_group({"video_name": "12345_2.mp4"}) == "12345"


def test_source_group_does_not_strip_internal_underscores():
    assert source_group({"video_name": "source_left.mp4"}) == "source_left"
