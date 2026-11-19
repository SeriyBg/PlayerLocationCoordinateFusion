from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import SAM, YOLO


IMAGE_PATH = Path(
    "data/spiideo/images/test/000017.jpg"
)

SAM3_MODEL_PATH = Path(
    "../models/sam3.pt"
)

OUTPUT_PATH = Path(
    "../output/000017_sam3_players.jpg"
)

MASK_DIRECTORY = Path(
    "../output/000017_masks"
)

# YOLO_MODEL = "yolo26n.pt"
YOLO_MODEL = "models/yolo26n_fullhd/yolo26n_fullhd-2/weights/best_original.pt"


def select_device() -> str | int:
    if torch.cuda.is_available():
        return 0

    mps_backend = getattr(torch.backends, "mps", None)

    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


def clip_boxes(
    boxes: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Ensure that all bbox coordinates are inside the image."""

    boxes = boxes.copy()

    boxes[:, [0, 2]] = np.clip(
        boxes[:, [0, 2]],
        0,
        image_width - 1,
    )

    boxes[:, [1, 3]] = np.clip(
        boxes[:, [1, 3]],
        0,
        image_height - 1,
    )

    return boxes


def resize_mask_to_image(
    mask: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Resize a SAM mask to the original image if necessary."""

    if mask.shape == (image_height, image_width):
        return mask.astype(bool)

    resized = cv2.resize(
        mask.astype(np.uint8),
        (image_width, image_height),
        interpolation=cv2.INTER_NEAREST,
    )

    return resized.astype(bool)


def find_mask_ground_point(
    mask: np.ndarray,
    bottom_band_height: int = 3,
) -> tuple[int, int] | None:
    """
    Find an approximate player ground-contact point.

    Rather than taking one extreme pixel, use the median x-coordinate
    of mask pixels within a small band at the bottom of the mask.
    """

    ys, xs = np.nonzero(mask)

    if len(xs) == 0:
        return None

    lowest_y = int(ys.max())

    bottom_pixels = ys >= lowest_y - bottom_band_height
    bottom_xs = xs[bottom_pixels]
    bottom_ys = ys[bottom_pixels]

    ground_x = int(np.median(bottom_xs))
    ground_y = int(np.median(bottom_ys))

    return ground_x, ground_y


def main() -> None:
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    if not SAM3_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"SAM3 checkpoint not found: {SAM3_MODEL_PATH}"
        )

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise RuntimeError(f"Could not read image: {IMAGE_PATH}")

    image_height, image_width = image.shape[:2]
    device = select_device()

    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # 1. Detect people with YOLO
    # ---------------------------------------------------------

    yolo_model = YOLO(YOLO_MODEL)

    yolo_result = yolo_model.predict(
        source=image,
        classes=[0],       # COCO class 0 = person
        conf=0.20,
        iou=0.70,
        imgsz=1280,
        device=device,
        verbose=False,
    )[0]

    if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
        print("No people were detected.")
        return

    yolo_boxes = (
        yolo_result.boxes.xyxy
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    yolo_confidences = (
        yolo_result.boxes.conf
        .detach()
        .cpu()
        .numpy()
    )

    yolo_boxes = clip_boxes(
        boxes=yolo_boxes,
        image_width=image_width,
        image_height=image_height,
    )

    print(f"YOLO detected {len(yolo_boxes)} people.")

    # Remove the YOLO model before loading SAM3.
    # This is useful on Macs with limited unified memory.
    del yolo_model
    del yolo_result

    if device == "mps":
        torch.mps.empty_cache()

    # ---------------------------------------------------------
    # 2. Segment every YOLO bbox using SAM3
    # ---------------------------------------------------------

    # sam_model = SAM(str(SAM3_MODEL_PATH))
    sam_model = SAM("mobile_sam.pt")

    sam_results = sam_model.predict(
        source=image,
        bboxes=yolo_boxes.tolist(),
        device=device,
        verbose=False,
        save=False,
    )

    if not sam_results:
        raise RuntimeError("SAM3 returned no results.")

    sam_result = sam_results[0]

    if sam_result.masks is None:
        print("SAM3 did not produce any masks.")
        return

    masks = (
        sam_result.masks.data
        .detach()
        .cpu()
        .numpy()
    )

    print(f"SAM3 generated {len(masks)} masks.")

    if len(masks) != len(yolo_boxes):
        print(
            "Warning: the number of SAM masks differs from "
            "the number of YOLO boxes."
        )

    # SAM/Ultralytics creates the basic segmentation visualization.
    annotated_image = sam_result.plot()

    MASK_DIRECTORY.mkdir(parents=True, exist_ok=True)

    number_of_pairs = min(len(masks), len(yolo_boxes))

    # ---------------------------------------------------------
    # 3. Save masks and calculate ground points
    # ---------------------------------------------------------

    for index in range(number_of_pairs):
        mask = resize_mask_to_image(
            mask=masks[index],
            image_width=image_width,
            image_height=image_height,
        )

        bbox = yolo_boxes[index]
        confidence = float(yolo_confidences[index])

        binary_mask = mask.astype(np.uint8) * 255

        mask_path = MASK_DIRECTORY / f"player_{index + 1:03d}.png"

        success = cv2.imwrite(
            str(mask_path),
            binary_mask,
        )

        if not success:
            raise RuntimeError(
                f"Could not save mask: {mask_path}"
            )

        ground_point = find_mask_ground_point(mask)

        if ground_point is not None:
            ground_x, ground_y = ground_point

            cv2.circle(
                annotated_image,
                (ground_x, ground_y),
                7,
                (0, 0, 255),
                -1,
            )

            cv2.putText(
                annotated_image,
                f"{index + 1}",
                (ground_x + 8, ground_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            print(
                f"Player {index + 1}: "
                f"YOLO confidence={confidence:.3f}, "
                f"ground point=({ground_x}, {ground_y})"
            )

        else:
            x1, y1, x2, y2 = bbox

            fallback_x = round((x1 + x2) / 2)
            fallback_y = round(y2)

            print(
                f"Player {index + 1}: empty mask; "
                f"bbox fallback=({fallback_x}, {fallback_y})"
            )

    # ---------------------------------------------------------
    # 4. Save and display the result
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(OUTPUT_PATH),
        annotated_image,
    )

    if not success:
        raise RuntimeError(
            f"Could not save output image: {OUTPUT_PATH}"
        )

    print(f"Visualization saved to: {OUTPUT_PATH}")
    print(f"Individual masks saved to: {MASK_DIRECTORY}")

    cv2.imshow(
        "YOLO detections segmented with SAM3",
        annotated_image,
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()