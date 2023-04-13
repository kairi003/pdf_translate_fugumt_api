from reportlab.pdfgen import canvas
from reportlab.platypus.frames import Frame
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, KeepInFrame
import io
from pypdf import PdfWriter, PdfReader
from transformers import pipeline, Pipeline
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from PIL import ImageFont

import layoutparser as lp

import matplotlib.pyplot as plt
import numpy as np
import pdf2image
from pathlib import Path
from logging import getLogger

logger = getLogger(__name__)

DPI = 72

# フォント登録
# https://github.com/googlefonts/morisawa-biz-ud-gothic/raw/main/fonts/ttf/BIZUDGothic-Regular.ttf
font_name = 'BIZUDGothic'
font_ttf = 'fonts/BIZUDGothic-Regular.ttf'
pdfmetrics.registerFont(TTFont(font_name, font_ttf))


# 特定のtext_blockがparagraph_blockに含まれているかチェック
def is_inside(paragraph_block, text_block):
    paragraph_width = paragraph_block.block.x_2 - paragraph_block.block.x_1
    paragraph_height = paragraph_block.block.y_2 - paragraph_block.block.y_1
    if paragraph_width > 300:
        allowable_error_pixel = 10
        return (text_block.block.x_1 >= paragraph_block.block.x_1 - allowable_error_pixel and text_block.block.y_1 >= paragraph_block.block.y_1 and
                text_block.block.x_2 <= paragraph_block.block.x_2 + allowable_error_pixel and text_block.block.y_2 <= paragraph_block.block.y_2 + allowable_error_pixel)
    else:
        allowable_error_pixel = 3
        return (text_block.block.x_1 >= paragraph_block.block.x_1 - allowable_error_pixel and text_block.block.y_1 >= paragraph_block.block.y_1 and
                text_block.block.x_2 <= paragraph_block.block.x_2 + allowable_error_pixel and text_block.block.y_2 <= paragraph_block.block.y_2 + allowable_error_pixel)


def fill_cover(canvas: canvas.Canvas, x, y, width, height):
    canvas.setFillColorRGB(1, 1, 1)
    # でかいパラグラフは検出精度悪いので補正する
    if width > 300:
        canvas.rect(
            x - 5,
            y,
            width + 10,
            height + 10,
            stroke=0,
            fill=1
        )
    else:
        canvas.rect(
            x,
            y,
            width,
            height,
            stroke=0,
            fill=1
        )


def calc_fontsize(paragraph_width, paragraph_height, translated_text):
    return int(np.sqrt((paragraph_width) * (paragraph_height) / len(translated_text)))


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


def run(target_pdf_file_path: Path, dpi: int, model: lp.Detectron2LayoutModel, translator: Pipeline, is_mihiraki: bool, font_name: str):

    # 翻訳モデル: fugumt

    # reportlab用の座標取る
    base_pdf = PdfReader(target_pdf_file_path)
    _, _, base_width, base_height = base_pdf.pages[0].mediabox

    output = PdfWriter()

    pdf_pages, pdf_images = lp.load_pdf(target_pdf_file_path, load_images=True, dpi=dpi)

    for page_index, (pdf_page, pdf_image) in enumerate(zip(pdf_pages, pdf_images)):
        logger.info(f'{page_index}')
        # テキストブロックを取得
        text_blocks: list[lp.TextBlock] = pdf_page.get_homogeneous_blocks()
        width, height = pdf_image.size
        # print(base_height, height, dpi)
        # レイアウトを取得
        pdf_layout = model.detect(pdf_image)
        # 段落ブロックの処理
        # 段落ブロックを抽出
        paragraph_blocks = lp.Layout(
            [b for b in pdf_layout if isinstance(b, lp.TextBlock) and b.type == 'Text'])

        cover_packet = io.BytesIO()
        cover_canvas = canvas.Canvas(cover_packet, pagesize=(
            int(base_width), int(base_height)), bottomup=True)

        text_packet = io.BytesIO()
        text_canvas = canvas.Canvas(text_packet, pagesize=(
            int(base_width), int(base_height)), bottomup=True)
        for paragraph_block in paragraph_blocks:
            inner_text_blocks = [
                text_block for text_block in text_blocks
                if text_block.is_in(
                    paragraph_block,
                    {'left': (p := 10 if paragraph_block.width > 300 else 3),
                     'right': p, 'bottom': p})
            ]
            # 段落中のテキストブロックを抽出
            print(len(inner_text_blocks))
            if len(inner_text_blocks) == 0:
                continue
            # 段落中のテキストブロックからテキストを抽出
            text = ' '.join(map(lambda x: x.text, inner_text_blocks))
            #print(text)
            translated_text = ''
            # テキストを翻訳
            if len(text) > 1000:
                n = len(text)
                i1 = n // 3
                i2 = i1 * 2

                text1 = text[:i1]
                text2 = text[i1:i2]
                text3 = text[i2:]

                result1 = translator(text1)
                result2 = translator(text2)
                result3 = translator(text3)
                translated_text = result1[0]['translation_text'] + \
                    result2[0]['translation_text'] + \
                    result3[0]['translation_text']
            else:
                result = translator(text)
                translated_text = result[0]['translation_text']
            #print(translated_text)
            paragraph_x = (paragraph_block.block.x_1 / width) * base_width
            paragraph_y = (paragraph_block.block.y_2 / height) * base_height
            paragraph_width = (
                (paragraph_block.block.width) / width) * base_width
            paragraph_height = (
                (paragraph_block.block.height) / height) * base_height

            # カバーフレームの追加
            fill_cover(cover_canvas, paragraph_x, base_height -
                       paragraph_y, paragraph_width, paragraph_height)

            # テキストフレームの追加
            frame = Frame(paragraph_x, base_height - paragraph_y, paragraph_width, paragraph_height,
                          showBoundary=0, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
            # テキスト実態の追加
            fontsize = calc_fontsize(
                paragraph_width, paragraph_height, translated_text)
            style = ParagraphStyle(
                name='Normal', fontName=font_name, fontSize=fontsize, leading=fontsize)
            paragraph = Paragraph(translated_text, style)
            story = [paragraph]
            story_inframe = KeepInFrame(
                paragraph_width * 1.5, paragraph_height * 1.5, story)
            frame.addFromList([story_inframe], text_canvas)

        # カバーをpdfページにする
        cover_canvas.save()

        cover_packet.seek(0)
        cover_pdf = PdfReader(cover_packet)

        # テキストをpdfページにする

        text_canvas.save()

        text_packet.seek(0)
        text_pdf = PdfReader(text_packet)

        # pdfをマージ
        base_pdf = PdfReader(open(target_pdf_file_path, "rb"))
        base_page = base_pdf.pages[page_index]
        # 見開き用
        if is_mihiraki:
            output.add_page(base_page)
        try:
            base_page.merge_page(cover_pdf.pages[0])
            base_page.merge_page(text_pdf.pages[0])
        except Exception as e:
            print("error: %s" % e)

        output.add_page(base_page)

    output_filepath = target_pdf_file_path.with_name("translated_" + target_pdf_file_path.name)
    output.write(output_filepath)


if __name__ == '__main__':
    # レイアウト(物体)検出モデルを準備
    target_pdf_file_path = Path('input.pdf')
    model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                                     extra_config=[
                                         "MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                                     label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"})
    translator = pipeline('translation', model='staka/fugumt-en-ja')
    is_mihiraki = True
    font_name = 'BIZUDGothic'
    run(target_pdf_file_path, DPI, model, translator, is_mihiraki, font_name)
