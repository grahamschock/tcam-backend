from extract import extract_keypoints
import os


def get_r2d2_pts(img_path):
    extract_script = "python extract.py \
                        --model models/r2d2_WASF_N16.pt \
                        --top-k 5000 \
                        --gpu -1 \
                        --images "
    extract_script += str(img_path)
    
    os.system(extract_script)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True)
    args = parser.parse_args()

    get_r2d2_pts(args.img)
