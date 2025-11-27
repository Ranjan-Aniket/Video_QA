"""
OCR Processor - Text Detection & Extraction from Frames

Model: PaddleOCR (300MB)
Accuracy: 92% clean text recognition
Purpose: Extract text from video frames with cost optimization
Compliance: JIT extraction per question, minimize OCR costs
Architecture: Evidence-first, supports batch processing

Primary Model: PaddleOCR (300MB, 92% accuracy on clean text)
Fallback: EasyOCR (free, local)
Advanced: Google Vision API for complex/low-quality text ($1.50/1000 images)
Target: ~$0.40 per video for OCR processing
"""

import logging
from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TextBlock:
    """Single detected text block with metadata"""
    text: str  # Detected text
    confidence: float  # Detection confidence (0.0-1.0)
    bounding_box: List[Tuple[int, int]]  # [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    language: Optional[str] = None  # Detected language
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of bounding box"""
        x_coords = [p[0] for p in self.bounding_box]
        y_coords = [p[1] for p in self.bounding_box]
        return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))
    
    @property
    def area(self) -> float:
        """Calculate approximate area of bounding box"""
        xs = [p[0] for p in self.bounding_box]
        ys = [p[1] for p in self.bounding_box]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        return width * height
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bounding_box": self.bounding_box,
            "language": self.language,
            "center": self.center,
            "area": self.area
        }


@dataclass
class FrameOCRResult:
    """OCR result for a single frame"""
    frame_index: int
    timestamp: float
    text_blocks: List[TextBlock]
    
    @property
    def all_text(self) -> str:
        """Concatenate all detected text"""
        return " ".join(block.text for block in self.text_blocks)
    
    @property
    def block_count(self) -> int:
        """Number of text blocks detected"""
        return len(self.text_blocks)
    
    @property
    def unique_words(self) -> Set[str]:
        """Get unique words from all text blocks"""
        words = set()
        for block in self.text_blocks:
            words.update(block.text.split())
        return words


@dataclass
class OCRExtractionResult:
    """Result of OCR extraction across multiple frames"""
    video_id: str
    frame_results: List[FrameOCRResult]
    total_text_blocks: int
    unique_words: Set[str]
    ocr_cost: float
    
    @property
    def frame_count(self) -> int:
        """Number of frames processed"""
        return len(self.frame_results)
    
    def get_text_at_timestamp(self, timestamp: float) -> str:
        """Get all text detected at specific timestamp"""
        for result in self.frame_results:
            if abs(result.timestamp - timestamp) < 0.5:  # 0.5s tolerance
                return result.all_text
        return ""


class OCRProcessor:
    """
    Extract text from video frames using OCR.
    
    Optimized for cost-effective processing:
    1. Use local EasyOCR by default (free)
    2. Fall back to Google Vision API for poor quality
    3. Batch process frames when possible
    4. Cache OCR results
    """
    
    def __init__(
        self,
        use_local_ocr: bool = True,
        google_vision_api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        enable_caching: bool = True
    ):
        """
        Initialize OCR processor.
        
        Args:
            use_local_ocr: Use EasyOCR (free) vs Google Vision API (paid)
            google_vision_api_key: API key for Google Vision (if used)
            cache_dir: Directory for caching OCR results
            enable_caching: Whether to cache results
        """
        self.use_local_ocr = use_local_ocr
        self.google_vision_api_key = google_vision_api_key
        self.cache_dir = cache_dir or Path("./cache/ocr")
        self.enable_caching = enable_caching
        
        # Initialize OCR engine
        self.ocr_engine = None
        if use_local_ocr:
            self._init_easyocr()
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"OCRProcessor initialized (local: {use_local_ocr})"
        )

    def _init_easyocr(self):
        """Initialize EasyOCR engine (lazy loading)"""
        try:
            import easyocr

            # Initialize EasyOCR with English language
            self.ocr_engine = easyocr.Reader(['en'], gpu=False, verbose=False)
            logger.info("EasyOCR engine initialized")
        except ImportError:
            logger.warning("EasyOCR not available. Install with: pip install easyocr")
            # Fall back to PaddleOCR
            self._init_paddleocr()
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            # Fall back to PaddleOCR
            self._init_paddleocr()

    def _init_paddleocr(self):
        """Initialize PaddleOCR engine (lazy loading) - FAST GPU MODE"""
        try:
            from paddleocr import PaddleOCR

            # Auto-detect GPU availability for PaddleOCR
            use_gpu = False
            try:
                import torch
                use_gpu = torch.cuda.is_available()
            except ImportError:
                pass

            # ⚡ FAST MODE: Optimized config for maximum speed with GPU
            ocr_config = {
                'use_angle_cls': False,  # ⚡ Disable angle classification (faster)
                'lang': 'en',  # English only
                'show_log': False,  # Suppress logging
                'use_gpu': use_gpu,  # GPU acceleration (10x faster)
                'enable_mkldnn': not use_gpu,  # Intel MKL-DNN only on CPU
                'det_db_thresh': 0.3,  # ⚡ Lower threshold for faster detection
                'det_db_box_thresh': 0.5,  # ⚡ Lower box threshold
                'use_mp': True,  # ⚡ Enable multiprocessing for batch processing
                'total_process_num': 4,  # ⚡ Use 4 processes for parallel processing
                'use_dilation': False,  # ⚡ Disable dilation (faster)
                'det_db_unclip_ratio': 1.5,  # Default unclip ratio
            }

            # GPU-specific optimizations
            if use_gpu:
                ocr_config.update({
                    'gpu_mem': 4000,  # ⚡ Allocate 4GB GPU memory (more = faster)
                    'use_tensorrt': False,  # TensorRT optimization (requires separate install)
                    'precision': 'fp16',  # ⚡ FP16 precision for faster inference on GPU
                })
                logger.info("⚡ PaddleOCR FAST GPU MODE: FP16 precision, 4GB VRAM, multiprocessing enabled")
            else:
                logger.info("⚡ PaddleOCR FAST CPU MODE: Multiprocessing enabled")

            self.ocr_engine = PaddleOCR(**ocr_config)
            logger.info(f"✓ PaddleOCR ready ({'GPU' if use_gpu else 'CPU'} mode)")
        except ImportError:
            logger.warning("PaddleOCR not available. Install with: pip install paddleocr paddlepaddle-gpu")
            self.ocr_engine = None
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            logger.info("Retrying with basic config...")
            try:
                # Fallback to basic config
                self.ocr_engine = PaddleOCR(
                    use_angle_cls=False,
                    lang='en',
                    show_log=False,
                    use_gpu=use_gpu
                )
                logger.info(f"✓ PaddleOCR ready (basic config)")
            except:
                self.ocr_engine = None
    
    def extract_text_from_frames(
        self,
        frames: List[np.ndarray],
        timestamps: List[float],
        video_id: str
    ) -> OCRExtractionResult:
        """
        Extract text from multiple frames.
        
        Args:
            frames: List of frame images (HxWxC numpy arrays)
            timestamps: Timestamp for each frame
            video_id: Unique video identifier
        
        Returns:
            OCRExtractionResult with detected text
        """
        logger.info(f"Extracting text from {len(frames)} frames")
        
        frame_results = []
        all_unique_words = set()
        total_blocks = 0
        ocr_cost = 0.0
        
        for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
            # Extract text from single frame
            text_blocks = self._extract_text_from_frame(frame)
            
            result = FrameOCRResult(
                frame_index=i,
                timestamp=timestamp,
                text_blocks=text_blocks
            )
            
            frame_results.append(result)
            all_unique_words.update(result.unique_words)
            total_blocks += len(text_blocks)
            
            # Calculate cost
            if self.use_local_ocr:
                ocr_cost += 0.0  # EasyOCR is free
            else:
                ocr_cost += 0.0015  # Google Vision: $1.50/1000 images
        
        extraction_result = OCRExtractionResult(
            video_id=video_id,
            frame_results=frame_results,
            total_text_blocks=total_blocks,
            unique_words=all_unique_words,
            ocr_cost=ocr_cost
        )
        
        logger.info(
            f"Extracted {total_blocks} text blocks, "
            f"{len(all_unique_words)} unique words "
            f"(cost: ${ocr_cost:.4f})"
        )
        
        return extraction_result
    
    def extract_text_jit(
        self, frame: np.ndarray, timestamp: float
    ) -> FrameOCRResult:
        """
        Extract text from single frame on-demand (JIT).

        This is the most cost-effective method for question-specific OCR.

        Args:
            frame: Frame image (HxWxC numpy array)
            timestamp: Frame timestamp in video

        Returns:
            FrameOCRResult with detected text
        """
        text_blocks = self._extract_text_from_frame(frame)

        return FrameOCRResult(
            frame_index=-1,  # Unknown frame index for JIT
            timestamp=timestamp,
            text_blocks=text_blocks
        )

    def extract_text_from_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract text from single frame (public API for evidence extraction).

        Args:
            frame: Frame image (HxWxC numpy array)

        Returns:
            List of text block dictionaries with 'text', 'confidence', 'bounding_box', etc.
        """
        text_blocks = self._extract_text_from_frame(frame)

        # Convert TextBlock objects to dictionaries
        return [block.to_dict() for block in text_blocks]
    
    def _extract_text_from_frame(
        self, frame: np.ndarray
    ) -> List[TextBlock]:
        """
        Extract text from single frame using configured OCR engine.

        Args:
            frame: Frame image (HxWxC numpy array, RGB or BGR format)

        Returns:
            List of detected text blocks
        """
        if self.use_local_ocr:
            return self._extract_with_local_ocr(frame)
        else:
            return self._extract_with_google_vision(frame)

    def _extract_with_local_ocr(
        self, frame: np.ndarray
    ) -> List[TextBlock]:
        """
        Extract text using local OCR (EasyOCR or PaddleOCR).

        Args:
            frame: Frame image (HxWxC numpy array, BGR or RGB format)

        Returns:
            List of detected text blocks
        """
        # Initialize OCR engine if not already loaded
        if self.ocr_engine is None:
            logger.info("OCR engine not loaded, initializing...")
            self._init_easyocr()  # This will fallback to PaddleOCR if EasyOCR fails

        if self.ocr_engine is None:
            logger.warning("OCR engine not available, returning empty")
            return []

        try:
            # Check which OCR engine is loaded and use appropriate API
            import easyocr

            if isinstance(self.ocr_engine, easyocr.Reader):
                # EasyOCR API: readtext()
                # Returns: [([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], text, confidence), ...]
                results = self.ocr_engine.readtext(frame)

                if not results:
                    return []

                # Convert EasyOCR results to TextBlock objects
                text_blocks = []
                for bbox, text, confidence in results:
                    # bbox is already a list of [x, y] pairs
                    bbox_tuples = [(int(point[0]), int(point[1])) for point in bbox]

                    text_blocks.append(TextBlock(
                        text=text,
                        confidence=float(confidence),
                        bounding_box=bbox_tuples,
                        language="en"
                    ))
            else:
                # PaddleOCR API: ocr()
                # Returns: [[[bbox], (text, confidence)], ...]
                results = self.ocr_engine.ocr(frame, cls=True)

                if results is None or len(results) == 0:
                    return []

                # Convert PaddleOCR results to TextBlock objects
                text_blocks = []
                for line in results[0] if results[0] is not None else []:
                    if line is None:
                        continue

                    bbox, (text, confidence) = line

                    # bbox format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    bbox_tuples = [(int(point[0]), int(point[1])) for point in bbox]

                    text_blocks.append(TextBlock(
                        text=text,
                        confidence=float(confidence),
                        bounding_box=bbox_tuples,
                        language="en"
                    ))

            engine_name = "EasyOCR" if isinstance(self.ocr_engine, easyocr.Reader) else "PaddleOCR"
            logger.debug(f"{engine_name} detected {len(text_blocks)} text blocks")
            return text_blocks

        except Exception as e:
            engine_name = "EasyOCR/PaddleOCR"
            try:
                import easyocr
                if isinstance(self.ocr_engine, easyocr.Reader):
                    engine_name = "EasyOCR"
                else:
                    engine_name = "PaddleOCR"
            except:
                pass
            logger.error(f"{engine_name} extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _extract_with_google_vision(
        self, frame: np.ndarray
    ) -> List[TextBlock]:
        """
        Extract text using Google Vision API (cloud, paid).
        
        Args:
            frame: Frame image (HxWxC numpy array)
        
        Returns:
            List of detected text blocks
        """
        # TODO: Implement with Google Vision API
        # from google.cloud import vision
        # import io
        # from PIL import Image
        # 
        # client = vision.ImageAnnotatorClient(
        #     credentials=...  # Use API key
        # )
        # 
        # # Convert numpy array to bytes
        # pil_image = Image.fromarray(frame)
        # img_byte_arr = io.BytesIO()
        # pil_image.save(img_byte_arr, format='PNG')
        # img_byte_arr = img_byte_arr.getvalue()
        # 
        # image = vision.Image(content=img_byte_arr)
        # 
        # # Detect text
        # response = client.text_detection(image=image)
        # texts = response.text_annotations
        # 
        # # Convert to TextBlock objects
        # text_blocks = []
        # for text in texts[1:]:  # Skip first (full text)
        #     vertices = [(v.x, v.y) for v in text.bounding_poly.vertices]
        #     text_blocks.append(TextBlock(
        #         text=text.description,
        #         confidence=1.0,  # Google Vision doesn't provide confidence
        #         bounding_box=vertices,
        #         language=text.locale if hasattr(text, 'locale') else None
        #     ))
        # 
        # return text_blocks
        
        logger.warning("_extract_with_google_vision not implemented - placeholder")
        return []
    
    def filter_text_blocks(
        self,
        text_blocks: List[TextBlock],
        min_confidence: float = 0.5,
        min_text_length: int = 2
    ) -> List[TextBlock]:
        """
        Filter text blocks by confidence and length.
        
        Args:
            text_blocks: List of text blocks to filter
            min_confidence: Minimum confidence threshold
            min_text_length: Minimum text length (characters)
        
        Returns:
            Filtered list of text blocks
        """
        filtered = [
            block for block in text_blocks
            if block.confidence >= min_confidence
            and len(block.text) >= min_text_length
        ]
        
        logger.debug(
            f"Filtered {len(text_blocks)} blocks to {len(filtered)} "
            f"(min_conf={min_confidence}, min_len={min_text_length})"
        )
        
        return filtered
    
    def search_text_in_results(
        self,
        results: OCRExtractionResult,
        search_term: str,
        case_sensitive: bool = False
    ) -> List[Tuple[float, str]]:
        """
        Search for specific text across all frames.
        
        Args:
            results: OCR extraction results
            search_term: Text to search for
            case_sensitive: Whether search is case-sensitive
        
        Returns:
            List of (timestamp, matching_text) tuples
        """
        matches = []
        
        for frame_result in results.frame_results:
            for block in frame_result.text_blocks:
                text = block.text
                search = search_term
                
                if not case_sensitive:
                    text = text.lower()
                    search = search.lower()
                
                if search in text:
                    matches.append((frame_result.timestamp, block.text))
        
        logger.debug(
            f"Found {len(matches)} matches for '{search_term}'"
        )
        
        return matches


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize processor with local OCR (free)
    processor = OCRProcessor(use_local_ocr=True, enable_caching=False)
    
    # Example 1: JIT single frame OCR (recommended)
    # frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    # result = processor.extract_text_jit(frame, timestamp=10.5)
    # 
    # print(f"✓ Extracted text from frame at 10.5s:")
    # print(f"  Blocks: {result.block_count}")
    # print(f"  Text: {result.all_text}")
    
    # Example 2: Batch frame processing
    # frames = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(5)]
    # timestamps = [5.0, 10.0, 15.0, 20.0, 25.0]
    # 
    # result = processor.extract_text_from_frames(
    #     frames=frames,
    #     timestamps=timestamps,
    #     video_id="vid_abc123"
    # )
    # 
    # print(f"\n✓ Extracted text from {result.frame_count} frames")
    # print(f"  Total blocks: {result.total_text_blocks}")
    # print(f"  Unique words: {len(result.unique_words)}")
    # print(f"  Cost: ${result.ocr_cost:.4f}")
    
    # Example 3: Search for specific text
    # matches = processor.search_text_in_results(
    #     results=result,
    #     search_term="Warning",
    #     case_sensitive=False
    # )
    # 
    # for timestamp, text in matches:
    #     print(f"  {timestamp:.1f}s: {text}")
    
    print("OCR processor ready (implementation pending)")
