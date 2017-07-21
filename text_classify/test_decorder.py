# -*- coding: utf-8 -*-

"""
@Time    : 7/17/17 7:34 PM
@Author  : wong
@E-Mail  : wmx3ng@gmail.com
@File    : test_decorder.py
@Software: PyCharm
@Description:
"""
from string import punctuation


def pipeline_wrapper(func):
    def to_lower(x):
        return x.lower()

    def remove_punc(x):
        for p in punctuation:
            x = x.replace(p, '')

        return x

    def wrapper(*args, **kwargs):
        x = to_lower(*args, **kwargs)
        x = remove_punc(x)
        return func(x)


@pipeline_wrapper
def tokenize_whitespace(inText):
    print(inText)
    return inText.split()


s = "string. With. Punctuation?"
print(tokenize_whitespace(s))
