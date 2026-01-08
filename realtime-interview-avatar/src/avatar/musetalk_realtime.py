"""
MuseTalk Real-time Avatar Service
Docker Linux 환경에서 mmpose를 사용한 실시간 립싱크
"""

import os
import sys
import copy
import queue
import threading
import time
from typing import Optional, Callable, Dict, Any, List, Tuple
from dataclasses import dataclass
import numpy as np
import cv2
import torch
from loguru import logger

# MuseTalk imports
try:
    from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
    from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
    from musetalk.utils.blending import get_image, get_image_blending, get_image_prepare_material
    from musetalk.utils.face_parsing import FaceParsing
    from musetalk.utils.audio_processor import AudioProcessor
    MUSETALK_AVAILABLE = True
except ImportError as e:
    logger.warning(f"MuseTalk not available: {e}")
    MUSETALK_AVAILABLE = False

# mmpose for Real-time mode
try:
    from mmpose.apis import inference_topdown
    from mmpose.apis import init_model as init_pose_model
    MMPOSE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"mmpose not available: {e}")
    MMPOSE_AVAILABLE = False


@dataclass
class AvatarConfig:
    """Avatar configuration"""
    # Model paths
    unet_model_path: str = "./models/musetalkV15/unet.pth"
    unet_config: str = "./models/musetalk/config.json"
    vae_type: str = "sd-vae"
    whisper_dir: str = "./models/whisper"

    # Video settings
    fps: int = 25
    batch_size: int = 8
    resolution: Tuple[int, int] = (512, 512)

    # Processing settings
    use_float16: bool = True
    version: str = "v15"
    parsing_mode: str = "jaw"
    extra_margin: int = 10

    # Face parsing
    left_cheek_width: int = 90
    right_cheek_width: int = 90

    # Silence handling
    silence_blend: bool = True
    silence_attenuation: float = 0.3

    # Device
    device: str = "cuda"
    gpu_id: int = 0


@dataclass
class VideoFrame:
    """Video frame with metadata"""
    frame: np.ndarray
    timestamp: float
    frame_index: int
    is_speaking: bool = True


class MuseTalkRealtimeAvatar:
    """
    Real-time MuseTalk Avatar using mmpose

    Producer-Consumer pattern for concurrent processing:
    - Producer: UNet inference (generates lip-synced faces)
    - Consumer: Post-processing (blending with original frame)
    """

    def __init__(
        self,
        source_image_path: str,
        config: Optional[AvatarConfig] = None,
        on_frame_ready: Optional[Callable[[VideoFrame], None]] = None,
    ):
        """
        Initialize Real-time Avatar

        Args:
            source_image_path: Path to source image/video
            config: Avatar configuration
            on_frame_ready: Callback when frame is ready
        """
        if not MUSETALK_AVAILABLE:
            raise ImportError("MuseTalk is required. Please install MuseTalk dependencies.")

        self.source_image_path = source_image_path
        self.config = config or AvatarConfig()
        self._on_frame_ready = on_frame_ready

        # Device setup
        self.device = torch.device(
            f"cuda:{self.config.gpu_id}"
            if torch.cuda.is_available() else "cpu"
        )

        # Models (lazy loaded)
        self._vae = None
        self._unet = None
        self._pe = None
        self._whisper = None
        self._audio_processor = None
        self._face_parser = None

        # Source frames and coordinates (pre-computed)
        self._frame_list = None
        self._coord_list = None
        self._input_latent_list = None
        self._mask_coords_list = None  # Pre-computed masks for Real-time
        self._mask_list = None

        # Processing state
        self._is_initialized = False
        self._is_processing = False
        self._frame_queue = queue.Queue(maxsize=100)
        self._result_queue = queue.Queue(maxsize=100)

        # Statistics
        self._total_frames = 0
        self._processing_times = []
        self._start_time = None

        # Threading
        self._process_thread = None
        self._stop_event = threading.Event()

        logger.info(f"MuseTalkRealtimeAvatar created: device={self.device}")

    def initialize(self) -> None:
        """Initialize models and preprocess source image"""
        if self._is_initialized:
            logger.warning("Avatar already initialized")
            return

        logger.info("Initializing MuseTalk Real-time Avatar...")
        start_time = time.time()

        # 1. Load models
        self._load_models()

        # 2. Preprocess source image/video
        self._preprocess_source()

        # 3. Pre-compute masks (Real-time key optimization)
        self._precompute_masks()

        self._is_initialized = True
        init_time = time.time() - start_time
        logger.info(f"Avatar initialized in {init_time:.2f}s")

    def _load_models(self) -> None:
        """Load all required models"""
        logger.info("Loading models...")

        # Load VAE, UNet, PositionalEncoding
        self._vae, self._unet, self._pe = load_all_model(
            unet_model_path=self.config.unet_model_path,
            vae_type=self.config.vae_type,
            unet_config=self.config.unet_config,
            device=self.device
        )

        # Convert to float16 if enabled
        if self.config.use_float16:
            self._pe = self._pe.half()
            self._vae.vae = self._vae.vae.half()
            self._unet.model = self._unet.model.half()

        # Move to device
        self._pe = self._pe.to(self.device)
        self._vae.vae = self._vae.vae.to(self.device)
        self._unet.model = self._unet.model.to(self.device)

        # Timesteps
        self._timesteps = torch.tensor([0], device=self.device)

        # Audio processor
        self._audio_processor = AudioProcessor(
            feature_extractor_path=self.config.whisper_dir
        )

        # Whisper model
        from transformers import WhisperModel
        weight_dtype = self._unet.model.dtype
        self._whisper = WhisperModel.from_pretrained(self.config.whisper_dir)
        self._whisper = self._whisper.to(device=self.device, dtype=weight_dtype).eval()
        self._whisper.requires_grad_(False)

        # Face parser
        if self.config.version == "v15":
            self._face_parser = FaceParsing(
                left_cheek_width=self.config.left_cheek_width,
                right_cheek_width=self.config.right_cheek_width
            )
        else:
            self._face_parser = FaceParsing()

        logger.info("Models loaded successfully")

    def _preprocess_source(self) -> None:
        """Preprocess source image/video and extract frames"""
        logger.info(f"Preprocessing source: {self.source_image_path}")

        file_type = get_file_type(self.source_image_path)

        if file_type == "video":
            # Extract frames from video
            import glob
            import tempfile

            temp_dir = tempfile.mkdtemp()
            cmd = f'ffmpeg -v fatal -i "{self.source_image_path}" -start_number 0 "{temp_dir}/%08d.png"'
            os.system(cmd)

            input_img_list = sorted(glob.glob(os.path.join(temp_dir, '*.[jpJP][pnPN]*[gG]')))
            self._source_fps = get_video_fps(self.source_image_path)

        elif file_type == "image":
            input_img_list = [self.source_image_path]
            self._source_fps = self.config.fps

        else:
            raise ValueError(f"Unsupported source type: {self.source_image_path}")

        # Get landmarks and bounding boxes
        logger.info("Extracting landmarks...")
        self._coord_list, self._frame_list = get_landmark_and_bbox(
            input_img_list,
            bbox_shift=0  # v15 uses fixed bbox_shift
        )

        # Create cyclic lists for looping
        self._frame_list_cycle = self._frame_list + self._frame_list[::-1]
        self._coord_list_cycle = self._coord_list + self._coord_list[::-1]

        # Pre-compute latents
        logger.info("Pre-computing latents...")
        self._input_latent_list = []
        for bbox, frame in zip(self._coord_list, self._frame_list):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            if self.config.version == "v15":
                y2 = y2 + self.config.extra_margin
                y2 = min(y2, frame.shape[0])
            crop_frame = frame[y1:y2, x1:x2]
            crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = self._vae.get_latents_for_unet(crop_frame)
            self._input_latent_list.append(latents)

        # Cyclic latent list
        self._input_latent_list_cycle = self._input_latent_list + self._input_latent_list[::-1]

        logger.info(f"Source preprocessed: {len(self._frame_list)} frames")

    def _precompute_masks(self) -> None:
        """
        Pre-compute face parsing masks (Real-time key optimization)
        This is what makes Real-time mode ~30fps vs ~8fps
        """
        logger.info("Pre-computing face parsing masks...")

        self._mask_coords_list = []
        self._mask_list = []

        for i, (bbox, frame) in enumerate(zip(self._coord_list_cycle, self._frame_list_cycle)):
            if bbox == coord_placeholder:
                self._mask_coords_list.append(None)
                self._mask_list.append(None)
                continue

            x1, y1, x2, y2 = bbox
            if self.config.version == "v15":
                y2 = y2 + self.config.extra_margin
                y2 = min(y2, frame.shape[0])

            # Pre-compute mask and crop_box
            mask, crop_box = get_image_prepare_material(
                frame, [x1, y1, x2, y2],
                mode=self.config.parsing_mode,
                fp=self._face_parser
            )

            self._mask_coords_list.append(crop_box)
            self._mask_list.append(mask)

        logger.info(f"Pre-computed {len(self._mask_list)} masks")

    @torch.no_grad()
    def process_audio(self, audio_path: str) -> List[VideoFrame]:
        """
        Process audio file and generate lip-synced video frames

        Args:
            audio_path: Path to audio file

        Returns:
            List of VideoFrame objects
        """
        if not self._is_initialized:
            self.initialize()

        logger.info(f"Processing audio: {audio_path}")
        start_time = time.time()

        # 1. Extract audio features
        whisper_input_features, librosa_length = self._audio_processor.get_audio_feature(audio_path)

        weight_dtype = self._unet.model.dtype
        whisper_chunks = self._audio_processor.get_whisper_chunk(
            whisper_input_features,
            self.device,
            weight_dtype,
            self._whisper,
            librosa_length,
            fps=self.config.fps,
            audio_padding_length_left=2,
            audio_padding_length_right=2,
        )

        video_num = len(whisper_chunks)
        logger.info(f"Audio frames: {video_num}")

        # 2. Calculate audio energy for silence detection
        import librosa
        y, sr = librosa.load(audio_path, sr=16000)
        samples_per_frame = sr // self.config.fps
        frame_energies = []
        for i in range(video_num):
            start = i * samples_per_frame
            end = min(start + samples_per_frame, len(y))
            if start >= len(y):
                frame_energies.append(0)
            else:
                segment = y[start:end]
                energy = np.sqrt(np.mean(segment ** 2)) if len(segment) > 0 else 0
                frame_energies.append(energy)

        mean_energy = np.mean(frame_energies) if frame_energies else 0
        silence_threshold = mean_energy * 0.1

        # 3. Batch UNet inference
        gen = datagen(
            whisper_chunks=whisper_chunks,
            vae_encode_latents=self._input_latent_list_cycle,
            batch_size=self.config.batch_size,
            delay_frame=0,
            device=self.device,
        )

        res_frame_list = []
        total_batches = int(np.ceil(float(video_num) / self.config.batch_size))

        from tqdm import tqdm
        for whisper_batch, latent_batch in tqdm(gen, total=total_batches, desc="UNet Inference"):
            audio_feature_batch = self._pe(whisper_batch)
            latent_batch = latent_batch.to(dtype=self._unet.model.dtype)

            pred_latents = self._unet.model(
                latent_batch,
                self._timesteps,
                encoder_hidden_states=audio_feature_batch
            ).sample

            recon = self._vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_list.append(res_frame)

        # 4. Post-processing with pre-computed masks (Real-time optimization)
        logger.info("Post-processing with pre-computed masks...")
        video_frames = []
        prev_combine_frame = None

        for i, res_frame in enumerate(tqdm(res_frame_list, desc="Blending")):
            idx = i % len(self._coord_list_cycle)
            bbox = self._coord_list_cycle[idx]
            ori_frame = copy.deepcopy(self._frame_list_cycle[idx])
            mask = self._mask_list[idx]
            crop_box = self._mask_coords_list[idx]

            x1, y1, x2, y2 = bbox
            if self.config.version == "v15":
                y2 = y2 + self.config.extra_margin
                y2 = min(y2, ori_frame.shape[0])

            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except:
                continue

            # Use pre-computed mask for fast blending (Real-time key!)
            if mask is not None and crop_box is not None:
                generated_frame = get_image_blending(
                    ori_frame, res_frame, [x1, y1, x2, y2],
                    mask, crop_box
                )
            else:
                # Fallback to regular blending
                generated_frame = get_image(
                    ori_frame, res_frame, [x1, y1, x2, y2],
                    mode=self.config.parsing_mode, fp=self._face_parser
                )

            # Silence handling (v3 attenuation)
            frame_energy = frame_energies[i] if i < len(frame_energies) else 0

            if self.config.silence_blend:
                if frame_energy < silence_threshold:
                    # Silence: attenuate UNet output
                    if prev_combine_frame is not None:
                        combine_frame = cv2.addWeighted(
                            prev_combine_frame.astype(np.float32), 1.0 - self.config.silence_attenuation,
                            generated_frame.astype(np.float32), self.config.silence_attenuation, 0
                        ).astype(np.uint8)
                    else:
                        combine_frame = generated_frame
                elif frame_energy < silence_threshold * 5:
                    # Transition: smooth blend
                    blend_ratio = (frame_energy - silence_threshold) / (silence_threshold * 4)
                    blend_ratio = max(0.0, min(1.0, blend_ratio))
                    if prev_combine_frame is not None:
                        combine_frame = cv2.addWeighted(
                            prev_combine_frame.astype(np.float32), 1.0 - blend_ratio,
                            generated_frame.astype(np.float32), blend_ratio, 0
                        ).astype(np.uint8)
                    else:
                        combine_frame = generated_frame
                else:
                    # Speech: full UNet output
                    combine_frame = generated_frame
            else:
                combine_frame = generated_frame

            prev_combine_frame = combine_frame.copy()

            # Create VideoFrame
            video_frame = VideoFrame(
                frame=combine_frame,
                timestamp=i / self.config.fps,
                frame_index=i,
                is_speaking=(frame_energy >= silence_threshold)
            )
            video_frames.append(video_frame)

            # Callback
            if self._on_frame_ready:
                self._on_frame_ready(video_frame)

        total_time = time.time() - start_time
        fps_achieved = len(video_frames) / total_time
        logger.info(f"Processing complete: {len(video_frames)} frames in {total_time:.2f}s ({fps_achieved:.1f} fps)")

        return video_frames

    def process_audio_realtime(
        self,
        audio_chunks: List[np.ndarray],
        sample_rate: int = 16000
    ) -> List[VideoFrame]:
        """
        Process audio chunks in real-time streaming mode

        Args:
            audio_chunks: List of audio chunks (numpy arrays)
            sample_rate: Audio sample rate

        Returns:
            List of VideoFrame objects
        """
        # TODO: Implement streaming audio processing
        # This would process audio in chunks as they arrive
        raise NotImplementedError("Streaming audio processing not yet implemented")

    def get_idle_frame(self) -> np.ndarray:
        """Get an idle frame (source image with neutral expression)"""
        if self._frame_list:
            return self._frame_list[0].copy()
        return np.zeros((512, 512, 3), dtype=np.uint8)

    def save_video(
        self,
        frames: List[VideoFrame],
        output_path: str,
        audio_path: Optional[str] = None
    ) -> str:
        """
        Save video frames to file

        Args:
            frames: List of VideoFrame objects
            output_path: Output video path
            audio_path: Optional audio to merge

        Returns:
            Path to saved video
        """
        import tempfile

        # Save frames to temp directory
        temp_dir = tempfile.mkdtemp()
        for i, vf in enumerate(frames):
            cv2.imwrite(f"{temp_dir}/{str(i).zfill(8)}.png", vf.frame)

        # Create video
        temp_video = f"{temp_dir}/temp_video.mp4"
        cmd = f'ffmpeg -y -v warning -r {self.config.fps} -f image2 -i "{temp_dir}/%08d.png" -vcodec libx264 -vf format=yuv420p -crf 18 "{temp_video}"'
        os.system(cmd)

        # Merge audio if provided
        if audio_path:
            cmd = f'ffmpeg -y -v warning -i "{audio_path}" -i "{temp_video}" "{output_path}"'
            os.system(cmd)
        else:
            import shutil
            shutil.move(temp_video, output_path)

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

        logger.info(f"Video saved to: {output_path}")
        return output_path

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "initialized": self._is_initialized,
            "total_frames": self._total_frames,
            "device": str(self.device),
            "source_frames": len(self._frame_list) if self._frame_list else 0,
            "masks_precomputed": len(self._mask_list) if self._mask_list else 0,
        }

    def reset(self) -> None:
        """Reset avatar state"""
        self._total_frames = 0
        self._processing_times = []
        self._start_time = None
        logger.info("Avatar state reset")


# Helper function
def create_realtime_avatar(
    source_image_path: str,
    device: str = "cuda",
    use_float16: bool = True,
    silence_blend: bool = True,
    on_frame_ready: Optional[Callable[[VideoFrame], None]] = None,
) -> MuseTalkRealtimeAvatar:
    """
    Create a Real-time MuseTalk Avatar

    Args:
        source_image_path: Path to source image/video
        device: Device to use (cuda/cpu)
        use_float16: Use float16 for faster inference
        silence_blend: Enable silence handling
        on_frame_ready: Callback when frame is ready

    Returns:
        MuseTalkRealtimeAvatar instance
    """
    config = AvatarConfig(
        device=device,
        use_float16=use_float16,
        silence_blend=silence_blend,
    )

    return MuseTalkRealtimeAvatar(
        source_image_path=source_image_path,
        config=config,
        on_frame_ready=on_frame_ready,
    )


# Test
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Source image path")
    parser.add_argument("--audio", required=True, help="Audio file path")
    parser.add_argument("--output", default="output.mp4", help="Output video path")
    args = parser.parse_args()

    # Create avatar
    avatar = create_realtime_avatar(args.source)

    # Initialize
    avatar.initialize()

    # Process audio
    frames = avatar.process_audio(args.audio)

    # Save video
    avatar.save_video(frames, args.output, args.audio)

    # Print stats
    print(avatar.get_stats())
