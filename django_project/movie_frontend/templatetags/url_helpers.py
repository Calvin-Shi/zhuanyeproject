# movie_frontend/templatetags/url_helpers.py

from django import template

register = template.Library()

@register.simple_tag
def url_params(query_dict, key_to_remove):
    """
    接收一个 QueryDict 和一个要移除的键名，
    返回一个移除了该键之后的新查询字符串。
    """
    params = query_dict.copy()
    if key_to_remove and key_to_remove in params:
        del params[key_to_remove]
    return params.urlencode()