import json
import pytest
from ca import glim_config as config


@pytest.fixture
def settings(tmp_path):
    (tmp_path/'config.json').write_text('{/* global */ "global": {"config_ros":"topics.json"}}')
    (tmp_path/'topics.json').write_text('''{
      // ROS sensor selection
      "glim_ros": {"points_topic":"/points", "imu_topic":"/raw",
        "note":"https://example.org/a/*literal*/", "quoted":"\\\"//literal"}
    }''')
    return tmp_path


def test_commented_json_and_literal_comment_markers(settings):
    result=config.inspect_config_topics(settings,'/points','/raw')
    assert result['status']=='matched'
    assert result['files'][1]['path'].endswith('topics.json')
    assert len(result['files'][0]['sha256'])==64


def test_defaults_follow_bag_reader(settings):
    (settings/'config.json').write_text('{}')
    (settings/'config_ros.json').write_text('{}')
    assert config.inspect_config_topics(settings,'/points','/imu')['status']=='matched'


@pytest.mark.parametrize('text', ['{"global":{},"global":{}}','{"global": NaN}',
    '{/* unfinished', '{"global":[],}', '[]'])
def test_invalid_json_rejected(settings,text):
    (settings/'config.json').write_text(text)
    with pytest.raises(ValueError): config.inspect_config_topics(settings,'/points','/raw')


@pytest.mark.parametrize('name',['../topics.json','/tmp/topics.json','a/../../topics.json','a\\topics.json',7])
def test_external_or_ambiguous_config_paths_rejected(settings,name):
    (settings/'config.json').write_text(json.dumps({'global':{'config_ros':name}}))
    with pytest.raises(ValueError,match='relative file'):
        config.inspect_config_topics(settings,'/points','/raw')


def test_topic_mismatch_and_missing_file(settings):
    with pytest.raises(ValueError,match='differ'):
        config.inspect_config_topics(settings,'/other','/raw')
    (settings/'topics.json').unlink()
    with pytest.raises(ValueError,match='regular file'):
        config.inspect_config_topics(settings,'/points','/raw')
