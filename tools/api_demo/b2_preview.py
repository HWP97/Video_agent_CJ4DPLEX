# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pydantic", "opencv-python-headless", "Pillow"]
# ///
# 실행: 기존 b1 Python -B tools/api_demo/b2_preview.py
"""대상 추적 초기 위치를 검토할 관측 종료 프레임 모음."""
import json
from PIL import Image, ImageDraw, ImageFont
import cv2
from b2_data import Case, OUT, VIDEO_ROOT


def main() -> None:
    cases = [Case.model_validate(r) for r in json.loads((OUT / "cases.json").read_text())]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    for start in range(0, len(cases), 8):
        sheet = Image.new("RGB", (1200, 1200), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, case in enumerate(cases[start:start+8]):
            cap = cv2.VideoCapture(str(VIDEO_ROOT / f"{case.video_id}.mp4"))
            cap.set(cv2.CAP_PROP_POS_FRAMES, round(case.cutoff_s*25))
            ok, frame = cap.read()
            cap.release()
            if not ok:
                continue
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = image.resize((600, 250))
            x, y = (offset % 2)*600, (offset//2)*300
            sheet.paste(image, (x,y+45))
            draw.text((x+4,y+2), f"{start+offset}: {case.sample_id}", font=font, fill="black")
            draw.text((x+4,y+23), case.target_description[:75], font=font, fill="black")
        sheet.save(OUT / f"target_review_{start//8}.jpg")


if __name__ == "__main__":
    main()
