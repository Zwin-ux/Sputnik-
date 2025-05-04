"""
Main script for running the HATD framework.
"""

import argparse
import logging
import os
import time
import sys
from src.pipeline.hatd_pipeline import HATDPipeline

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def process_input_file(pipeline, input_file, output_file=None):
    """
    Process an input file through the HATD pipeline.
    
    Args:
        pipeline: HATDPipeline instance
        input_file: Path to input file
        output_file: Path to output file (optional)
    """
    logger.info(f"Processing input file: {input_file}")
    
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Process through pipeline
    start_time = time.time()
    results = pipeline.process_input(text)
    processing_time = time.time() - start_time
    
    # Print results
    original_tokens = results["original_token_count"]
    processed_tokens = results["preprocessed_token_count"]
    token_reduction = results["token_reduction"]
    
    logger.info(f"Original tokens: {original_tokens}")
    logger.info(f"Processed tokens: {processed_tokens}")
    logger.info(f"Token reduction: {token_reduction:.2%}")
    logger.info(f"Processing time: {processing_time:.4f} seconds")
    logger.info(f"Selected model: {results['selected_model']}")
    
    # Save output if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# HATD Processed Text\n\n")
            f.write(f"## Original Text ({original_tokens} tokens)\n\n")
            f.write(results["original_text"])
            f.write(f"\n\n## Processed Text ({processed_tokens} tokens, {token_reduction:.2%} reduction)\n\n")
            f.write(results["preprocessed_text"])
            f.write(f"\n\n## Metrics\n\n")
            f.write(f"- Token reduction: {token_reduction:.2%}\n")
            f.write(f"- Preprocessing time: {results['preprocessing_time']:.4f} seconds\n")
            f.write(f"- Total processing time: {processing_time:.4f} seconds\n")
            f.write(f"- Selected model: {results['selected_model']}\n")
        
        logger.info(f"Output saved to: {output_file}")
    
    return results


def interactive_mode(pipeline):
    """
    Run the pipeline in interactive mode.
    
    Args:
        pipeline: HATDPipeline instance
    """
    logger.info("Starting interactive mode. Enter text to process (type 'exit' to quit).")
    
    while True:
        try:
            print("\nEnter text to process (or 'exit' to quit):")
            text = input("> ")
            
            if text.lower() == 'exit':
                break
            
            # Process input
            start_time = time.time()
            results = pipeline.process_input(text)
            processing_time = time.time() - start_time
            
            # Print results
            print("\n========== RESULTS ==========")
            print(f"Original text ({results['original_token_count']} tokens):")
            print(f"{text}")
            
            print(f"\nProcessed text ({results['preprocessed_token_count']} tokens, {results['token_reduction']:.2%} reduction):")
            print(f"{results['preprocessed_text']}")
            
            print(f"\nToken reduction: {results['token_reduction']:.2%}")
            print(f"Processing time: {processing_time:.4f} seconds")
            print(f"Selected model: {results['selected_model']}")
            print("==============================")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error processing input: {e}")
    
    logger.info("Interactive mode ended.")


def main():
    """
    Main function to run the HATD pipeline.
    """
    parser = argparse.ArgumentParser(description="Run HATD pipeline")
    parser.add_argument("--config", type=str, default="configs/default_config.json",
                       help="Path to configuration file")
    parser.add_argument("--input", type=str, default=None,
                       help="Path to input file")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to output file")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        return 1
    
    # Initialize pipeline
    logger.info(f"Initializing HATD pipeline with config: {args.config}")
    pipeline = HATDPipeline(config_path=args.config)
    
    # Run in appropriate mode
    if args.interactive:
        interactive_mode(pipeline)
    elif args.input:
        process_input_file(pipeline, args.input, args.output)
    else:
        logger.error("Either --input or --interactive must be specified")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
