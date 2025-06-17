import imageio.v2 as imageio
import os
from PIL import Image
import numpy as np


def create_video(input_folder, output_file, fps=30, codec='libx264'):
    episode_paths = os.listdir(input_folder)
    episode_paths.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)

    writer = imageio.get_writer(
        output_file,
        fps=fps,
        codec=codec,
        macro_block_size=8
    )

    for episode in episode_paths:
        episode_folder = os.path.join(input_folder, episode)
        # sort the images by time step
        images = [img for img in os.listdir(episode_folder) if img.endswith(".png")]
        images.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)

        first_image = np.array(Image.open(os.path.join(episode_folder, images[0])))
        height, width, _ = first_image.shape

        # every prediction lasts 1/6 seconds
        frames_per_image = fps // 3

        for img_name in images:
            img_path = os.path.join(episode_folder, img_name)
            image = imageio.imread(img_path)
            
            # 调整图像尺寸为能被16整除（视频编码要求）
            if image.shape[1] % 16 != 0 or image.shape[0] % 16 != 0:
                new_width = (image.shape[1] // 16) * 16
                new_height = (image.shape[0] // 16) * 16
                image = image[:new_height, :new_width]
            
            for _ in range(frames_per_image):
                writer.append_data(image)

    writer.close()


if __name__ == "__main__":
    input_folder = "./outputs/visual/0313_14.25.32_train_diffusion_transformer_hybrid_weighted_multi_gpu_pouring_sirius"
    output_file = os.path.join(input_folder, "video.mp4")
    
    create_video(input_folder, output_file)
    
    print(f"Generated video: {output_file}")