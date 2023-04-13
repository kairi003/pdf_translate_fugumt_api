#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections.abc import Iterable
import io
from pathlib import Path
from logging import getLogger

import layoutparser as lp
import numpy as np
from PIL import ImageFont, Image
from pypdf import PdfReader, PdfWriter, PageObject
from pypdf.generic import RectangleObject
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import KeepInFrame, Paragraph
from reportlab.platypus.frames import Frame
from transformers import pipeline, Pipeline
import pdf2image
import pysbd

logger = getLogger(__name__)

DPI = 72
font_name = 'BIZUDGothic'
font_ttf = 'fonts/BIZUDGothic-Regular.ttf'
pdfmetrics.registerFont(TTFont(font_name, font_ttf))


class PdfOverlayBase:
    def __init__(self, base_width: int, base_height: int):
        self.buffer = io.BytesIO()
        self.canvas = canvas.Canvas(self.buffer, pagesize=(
            base_width, base_height), bottomup=True)

    def merge(self, base: PageObject):
        self.canvas.save()
        self.buffer.seek(0)
        pdf = PdfReader(self.buffer)
        if len(pdf.pages) == 0:
            return
        base.merge_page(pdf.pages[0])


class PdfCoverOverlay(PdfOverlayBase):
    def fill(self, x, y, width, height):
        self.canvas.setFillColorRGB(1, 1, 1)
        # でかいパラグラフは検出精度悪いので補正する
        if width > 300:
            self.canvas.rect(
                x - 5,
                y,
                width + 10,
                height + 10,
                stroke=0,
                fill=1
            )
        else:
            self.canvas.rect(
                x,
                y,
                width,
                height,
                stroke=0,
                fill=1
            )


class PdfTextOverlay(PdfOverlayBase):
    def add_text(self, translated_text, x, y, width, height, font_name):
        # テキストフレームとテキスト実態の追加部分
        frame = Frame(x, y, width, height, showBoundary=0, leftPadding=0,
                      rightPadding=0, topPadding=0, bottomPadding=0)
        fontsize = PdfFontSizeCalculator.calc_fontsize(
            width, height, translated_text)
        style = ParagraphStyle(
            name='Normal', fontName=font_name, fontSize=fontsize, leading=fontsize)
        paragraph = Paragraph(translated_text, style)
        story = [paragraph]
        story_inframe = KeepInFrame(width * 1.5, height * 1.5, story)
        frame.addFromList([story_inframe], self.canvas)


class PdfFontSizeCalculator:
    @staticmethod
    def calc_fontsize(paragraph_width, paragraph_height, translated_text):
        return int(np.sqrt((paragraph_width) * (paragraph_height) / len(translated_text)))

    @staticmethod
    def get_max_font_size(paragraph_width, paragraph_height, translated_text, font_face="./BIZUDGothic-Regular.ttf", max_font_size=100):
        """
        指定された領域内で最大のフォントサイズを求める。
        :param text: 描画する文字列。
        :param font_face: フォント名。
        :param rectangle: 描画領域を表すタプル (x0, y0, x1, y1)。
        :param max_font_size: 最大フォントサイズ。デフォルトは 100。
        :return: 最大フォントサイズ。
        """
        for font_size in range(max_font_size, 0, -1):
            font = ImageFont.truetype(font_face, font_size)
            # 描画する文字列のサイズを求める
            text_width, text_height = font.getsize(translated_text)
            if text_width <= paragraph_width and text_height <= paragraph_height:
                return font_size
        return 0


class PDFLayoutExtractor:
    def __init__(self, model: lp.Detectron2LayoutModel, translator: Pipeline):
        self.model = model
        self.translator = translator

    def extract_layout(self, target_pdf_file_path: Path, base_pdf: PdfReader, dpi: int, font_name: str) -> list[list[lp.TextBlock]]:
        ...
        # レイアウトを抽出する部分のコード


class Translator:
    def __init__(self):
        self.translator = pipeline('translation', model='staka/fugumt-en-ja')
        self.segmenter = pysbd.Segmenter(language='en', clean=False)

    def translate(self, text: str) -> str:
        result = self.translator(self.segmenter.segment(text))
        translated_text = result[0]['translation_text']
        return translated_text
    """
        if len(text) > 1000:
            n = len(text)
            i1 = n // 3
            i2 = i1 * 2

            text1 = text[:i1]
            text2 = text[i1:i2]
            text3 = text[i2:]

            result1 = self.translator(text1)
            result2 = self.translator(text2)
            result3 = self.translator(text3)
            translated_text = result1[0]['translation_text'] + \
                result2[0]['translation_text'] + \
                result3[0]['translation_text']
        else:
            result = self.translator(text)
            translated_text = result[0]['translation_text']"""


class PDFTranslator:
    def __init__(self,
                 dpi: int,
                 model: lp.Detectron2LayoutModel,
                 translator: Translator,
                 is_mihiraki: bool,
                 font_name: str):
        self.dpi = dpi
        self.model = model
        self.translator = translator
        self.is_mihiraki = is_mihiraki
        self.font_name = font_name

    def one_page(self, base_page: PageObject, page_layout: lp.Layout, page_image: Image.Image):
        _, _, base_width, base_height = base_page.mediabox
        im_width, im_height = page_image.size

        pdf_layout = self.model.detect(page_image)
        paragraph_blocks = (b for b in pdf_layout if b.type == 'Text')
        
        text_blocks: list[lp.TextBlock] = page_layout.get_homogeneous_blocks()

        cover_overlay = PdfCoverOverlay(base_width, base_height)
        text_overlay = PdfTextOverlay(base_width, base_height)

        def is_in(inner: lp.Rectangle, outer: lp.Rectangle):
            m = 10 if outer.width > 300 else 3
            return inner.is_in(outer, {'left': m, 'right': m, 'bottom': m})

        for paragraph_block in paragraph_blocks:
            text = ' '.join(b.text for b in text_blocks if is_in(b, paragraph_block))
            if not text:
                continue
            translated_text = self.translator.translate(text)
            logger.info(translated_text)

            paragraph_x = paragraph_block.block.x_1 * base_width / im_width
            paragraph_y = paragraph_block.block.y_2 * base_height / im_height
            paragraph_width = paragraph_block.block.width * base_width / im_width
            paragraph_height = paragraph_block.block.height * base_height / im_height

            cover_overlay.fill(paragraph_x, base_height -
                               paragraph_y, paragraph_width, paragraph_height)
            text_overlay.add_text(translated_text, paragraph_x, base_height -
                                  paragraph_y, paragraph_width, paragraph_height, self.font_name)

        cover_overlay.merge(base_page)
        text_overlay.merge(base_page)
        return base_page

    def run(self, target_path: Path):
        base_pdf = PdfReader(target_path)
        page_layouts, page_images = lp.load_pdf(target_path, load_images=True, dpi=self.dpi)

        output = PdfWriter()

        for base, layout, image in zip(base_pdf.pages, page_layouts, page_images):
            translated_page = self.one_page(base, layout, image)
            output.add_page(translated_page)

        output_filepath = target_path.with_name("translated_" + target_path.name)
        output.write(output_filepath)
        return output


if __name__ == '__main__':
    ...
    # 以前と同じ__main__部分でPDFTranslatorのインスタンスを作成し、runメソッドを実行
