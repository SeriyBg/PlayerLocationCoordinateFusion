from pathlib import Path

import cv2

IMAGE_PATH = Path(
    "data/"
    "spiideo_yolo/images/train/000028.jpg"
)

LABEL_PATH = Path(
    "data/"
    "spiideo_yolo/labels/train/000028.txt"
)


def main() -> None:
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {IMAGE_PATH}"
        )

    image_height, image_width = image.shape[:2]

    for line in LABEL_PATH.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        class_id, x_center, y_center, width, height = map(
            float,
            line.split(),
        )

        x_center *= image_width
        y_center *= image_height
        width *= image_width
        height *= image_height

        x1 = round(x_center - width / 2)
        y1 = round(y_center - height / 2)
        x2 = round(x_center + width / 2)
        y2 = round(y_center + height / 2)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

    cv2.imshow("Converted SynLoc labels", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()