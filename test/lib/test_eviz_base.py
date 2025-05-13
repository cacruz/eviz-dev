def test_eviz_base_object_source_name_size(get_eviz):
    obj = get_eviz()
    assert len(obj.source_names) == 1


def test_eviz_base_object_source_name(get_eviz):
    obj = get_eviz()
    assert obj.source_names[0] == 'test'


def test_eviz_base_object_factory_sources(get_eviz):
    obj = get_eviz()
    assert isinstance(obj.factory_sources, list)

