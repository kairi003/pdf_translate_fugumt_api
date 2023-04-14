#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import io
import tempfile
from logging import getLogger
from pathlib import Path
from typing import Generator, Union

from PIL import Image
from pypdf import PdfReader, PdfWriter, PageObject
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from layoutparser.elements.layout import Layout
from layoutparser.io.pdf import load_pdf
from layoutparser.models.detectron2 import Detectron2LayoutModel

from .overlay import PdfOverlay, ParagraphPosition
from .translator import Translator
from .extractor import PdfLayoutExtractor


logger = getLogger(__name__)

DEFAULT_DPI = 72
font_name = 'BIZUDGothic'
font_ttf = 'fonts/BIZUDGothic-Regular.ttf'
pdfmetrics.registerFont(TTFont(font_name, font_ttf))


class PdfTranslator:
    def __init__(self,
                 dpi: int,
                 model: Detectron2LayoutModel,
                 translator: Translator,
                 is_mihiraki: bool,
                 font_name: str):
        self.dpi = dpi
        self.model = model
        self.translator = translator
        self.is_mihiraki = is_mihiraki
        self.font_name = font_name
        self.layout_extractor = PdfLayoutExtractor(self.model)

    def load_pdf(self, source: Union[Path, io.BytesIO]) \
            -> tuple[PdfReader, Layout, list[Image.Image]]:
        base_pdf = PdfReader(source)

        if isinstance(source, io.BytesIO):
            with tempfile.NamedTemporaryFile() as temp_file:
                temp_file.write(source.getvalue())
                temp_file.flush()
                file_path = temp_file.name
                page_layouts, page_images = load_pdf(
                    file_path, load_images=True, dpi=self.dpi)
        else:
            page_layouts, page_images = load_pdf(
                source, load_images=True, dpi=self.dpi)

        return base_pdf, page_layouts, page_images

    def run(self, source: Union[Path, io.BytesIO]) -> PdfWriter:
        writer = PdfWriter()
        for page in self.execute(source):
            writer.add_page(page)
        return writer

    def execute(self, source: Union[Path, io.BytesIO]) -> Generator[PageObject, None, None]:
        base_pdf, page_layouts, page_images = self.load_pdf(source)
        for base, layout, image in zip(base_pdf.pages, page_layouts, page_images):
            if self.is_mihiraki:
                yield base
            translated_page = self.execute_one_page(base, layout, image)
            yield translated_page
    

    def execute_one_page(self, base_page: PageObject, page_layout: Layout, page_image: Image.Image):
        base_page = copy.copy(base_page)
        overlay = PdfOverlay(base_page)

        for para_block, text in self.layout_extractor.extract_layout(page_layout, page_image):
            translated_text = self.translator.translate(text)

            logger.debug(translated_text)

            pos = ParagraphPosition.from_paragraph_block(
                para_block, base_page, page_image)

            overlay.add(pos, translated_text, self.font_name)

        overlay.merge(base_page)
        return base_page


if __name__ == '__main__':
    ...
    # 以前と同じ__main__部分でPDFTranslatorのインスタンスを作成し、runメソッドを実行
