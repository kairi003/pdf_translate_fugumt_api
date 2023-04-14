#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
from dataclasses import dataclass
from typing import Generator

import numpy as np
from PIL import Image
from pypdf import PdfReader, PageObject
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import KeepInFrame, Paragraph
from reportlab.platypus.frames import Frame

from layoutparser.elements.layout import Layout


@dataclass(frozen=True)
class ParagraphPosition:
    x: float
    y: float
    width: float
    height: float

    @classmethod
    def from_paragraph_block(
            cls,
            paragraph_block: Layout,
            base_page: PageObject,
            page_image: Image.Image):
        _, _, base_width, base_height = base_page.mediabox
        im_width, im_height = page_image.size
        x = paragraph_block.block.x_1 * base_width / im_width
        y = base_height - paragraph_block.block.y_2 * base_height / im_height
        width = paragraph_block.block.width * base_width / im_width
        height = paragraph_block.block.height * base_height / im_height
        return cls(x, y, width, height)

    def __iter__(self) -> Generator[float, None, None]:
        yield self.x
        yield self.y
        yield self.width
        yield self.height


class PdfOverlayBase:
    def __init__(self, base: PageObject):
        _, _, width, height = base.mediabox
        self.buffer = io.BytesIO()
        self.canvas = canvas.Canvas(
            self.buffer, pagesize=(width, height), bottomup=True)

    def merge(self, base: PageObject):
        self.canvas.save()
        self.buffer.seek(0)
        pdf = PdfReader(self.buffer)
        if len(pdf.pages) == 0:
            return
        base.merge_page(pdf.pages[0])
        return


class PdfCoverOverlay(PdfOverlayBase):
    def fill(self, pos: ParagraphPosition):
        self.canvas.setFillColorRGB(1, 1, 1)
        x, y, width, height = pos
        # Correct for large paragraphs with low detection accuracy.
        if width > 300:
            x -= 5
            width += 10
            height += 10
        self.canvas.rect(x, y, width, height, stroke=0, fill=1)


class PdfTextOverlay(PdfOverlayBase):
    def add_text(self, pos: ParagraphPosition, translated_text: str, font_name: str):
        x, y, width, height = pos
        frame = Frame(x, y, width, height, showBoundary=0, leftPadding=0,
                      rightPadding=0, topPadding=0, bottomPadding=0)
        fontsize = self.calc_fontsize(width, height, translated_text)
        style = ParagraphStyle(
            name='Normal', fontName=font_name, fontSize=fontsize, leading=fontsize)
        paragraph = Paragraph(translated_text, style)
        story = [paragraph]
        story_inframe = KeepInFrame(width * 1.5, height * 1.5, story)
        frame.addFromList([story_inframe], self.canvas)

    @staticmethod
    def calc_fontsize(paragraph_width, paragraph_height, translated_text):
        return int(np.sqrt((paragraph_width) * (paragraph_height) / len(translated_text)))


class PdfOverlay(PdfOverlayBase):
    def __init__(self, base: PageObject):
        super().__init__(base)
        self.cover_overlay = PdfCoverOverlay(base)
        self.text_overlay = PdfTextOverlay(base)

    def add(self, pos: ParagraphPosition, translated_text: str, font_name: str):
        self.cover_overlay.fill(pos)
        self.text_overlay.add_text(pos, translated_text, font_name)

    def merge(self, base: PageObject):
        self.cover_overlay.merge(base)
        self.text_overlay.merge(base)
        return
