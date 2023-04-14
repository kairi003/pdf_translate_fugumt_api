#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pysbd
from transformers import pipeline

class Translator:
    def __init__(self):
        self.translator = pipeline('translation', model='staka/fugumt-en-ja')
        self.segmenter = pysbd.Segmenter(language='en', clean=False)

    def translate(self, text: str) -> str:
        result = self.translator(self.segmenter.segment(text))
        translated_text = ' '.join(res['translation_text'] for res in result)
        return translated_text
