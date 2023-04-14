#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Generator
from PIL import Image

from layoutparser.elements.layout import Layout
from layoutparser.elements.layout_elements import TextBlock
from layoutparser.models.detectron2 import Detectron2LayoutModel


class PdfLayoutExtractor:

    def __init__(self, model: Detectron2LayoutModel):
        self.model = model

    def extract_layout(self, page_layout: Layout, page_image: Image.Image) -> Generator[tuple[TextBlock, str], None, None]:
        # Paragraph Blocks by detectron2
        detected_layout = self.model.detect(page_image)
        paragraph_blocks = [
            block for block in detected_layout if isinstance(block, TextBlock) and block.type == 'Text']

        # Text Blocks by LayoutParser
        text_blocks = [
            block for block in page_layout.get_homogeneous_blocks()
            if isinstance(block, TextBlock) and isinstance(block.text, str)]

        for paragraph_block in paragraph_blocks:
            text = ' '.join(
                block.text for block in text_blocks if self.is_in(block, paragraph_block))
            if not text:
                continue
            yield paragraph_block, text

    @staticmethod
    def is_in(inner: TextBlock, outer: TextBlock):
        m = 10 if outer.width > 300 else 3
        return inner.is_in(outer, {'left': m, 'right': m, 'bottom': m})
