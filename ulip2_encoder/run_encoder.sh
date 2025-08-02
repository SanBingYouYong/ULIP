#!/bin/bash

# ULIP2 Encoder Runner Script
# This script provides convenient commands to run the ULIP2 encoder

# Set default model path (when run from parent directory)
MODEL_PATH="ulip_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt"

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_help() {
    echo "ULIP2 Encoder Runner"
    echo "==================="
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  test              - Run test suite to verify installation"
    echo "  examples          - Run example demonstrations"
    echo "  encode-obj        - Encode a single OBJ file"
    echo "  encode-text       - Encode a text description"
    echo "  encode-dir        - Encode all files in a directory"
    echo "  compare           - Compare 3D model with text"
    echo "  build-db          - Build vector database from directory"
    echo "  search-db         - Search in vector database"
    echo "  download-model    - Download ULIP2 pretrained model"
    echo ""
    echo "Options:"
    echo "  --obj-file PATH   - Path to OBJ file"
    echo "  --obj-dir PATH    - Path to directory with 3D files"
    echo "  --text \"TEXT\"    - Text description"
    echo "  --output-dir PATH - Output directory (default: embeddings)"
    echo "  --db-path PATH    - Vector database path (default: vector_db.pkl)"
    echo "  --top-k N         - Number of search results (default: 5)"
    echo "  --device DEVICE   - Device to use: cuda or cpu (default: cuda)"
    echo "  --num-points N    - Number of points to sample (default: 10000)"
    echo ""
    echo "Examples:"
    echo "  $0 test"
    echo "  $0 encode-obj --obj-file chair.obj"
    echo "  $0 encode-text --text \"a red chair\""
    echo "  $0 compare --obj-file chair.obj --text \"a chair\""
    echo "  $0 build-db --obj-dir models/ --db-path furniture.pkl"
    echo "  $0 search-db --db-path furniture.pkl --text \"a table\""
}

check_model() {
    if [ ! -f "$MODEL_PATH" ]; then
        echo -e "${RED}Error: ULIP2 model not found at $MODEL_PATH${NC}"
        echo "Run: $0 download-model"
        return 1
    fi
    return 0
}

download_model() {
    echo -e "${YELLOW}Downloading ULIP2 pretrained model...${NC}"
    
    # Create directory if it doesn't exist
    mkdir -p "$(dirname "$MODEL_PATH")"
    
    # Download the model
    wget -O "$MODEL_PATH" \
        "https://huggingface.co/datasets/SFXX/ulip/resolve/main/ULIP-2/pretrained_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Model downloaded successfully!${NC}"
    else
        echo -e "${RED}Failed to download model${NC}"
        return 1
    fi
}

run_test() {
    echo -e "${YELLOW}Running ULIP2 encoder tests...${NC}"
    python test_encoder.py
}

run_examples() {
    echo -e "${YELLOW}Running ULIP2 encoder examples...${NC}"
    check_model || return 1
    python examples.py
}

# Parse command line arguments
COMMAND=""
OBJ_FILE=""
OBJ_DIR=""
TEXT=""
OUTPUT_DIR="embeddings"
DB_PATH="vector_db.pkl"
TOP_K=5
DEVICE="cuda"
NUM_POINTS=10000

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        test|examples|encode-obj|encode-text|encode-dir|compare|build-db|search-db|download-model)
            COMMAND="$1"
            shift
            ;;
        --obj-file)
            OBJ_FILE="$2"
            shift 2
            ;;
        --obj-dir)
            OBJ_DIR="$2"
            shift 2
            ;;
        --text)
            TEXT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --db-path)
            DB_PATH="$2"
            shift 2
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --num-points)
            NUM_POINTS="$2"
            shift 2
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_help
            exit 1
            ;;
    esac
done

# Execute command
case "$COMMAND" in
    "test")
        run_test
        ;;
    "examples")
        run_examples
        ;;
    "download-model")
        download_model
        ;;
    "encode-obj")
        if [ -z "$OBJ_FILE" ]; then
            echo -e "${RED}Error: --obj-file is required${NC}"
            exit 1
        fi
        check_model || exit 1
        echo -e "${YELLOW}Encoding OBJ file: $OBJ_FILE${NC}"
        python ulip2_encoder.py --obj_file "$OBJ_FILE" --output_dir "$OUTPUT_DIR" --device "$DEVICE" --num_points "$NUM_POINTS"
        ;;
    "encode-text")
        if [ -z "$TEXT" ]; then
            echo -e "${RED}Error: --text is required${NC}"
            exit 1
        fi
        check_model || exit 1
        echo -e "${YELLOW}Encoding text: $TEXT${NC}"
        python ulip2_encoder.py --text "$TEXT" --output_dir "$OUTPUT_DIR" --device "$DEVICE"
        ;;
    "encode-dir")
        if [ -z "$OBJ_DIR" ]; then
            echo -e "${RED}Error: --obj-dir is required${NC}"
            exit 1
        fi
        check_model || exit 1
        echo -e "${YELLOW}Encoding directory: $OBJ_DIR${NC}"
        python ulip2_encoder.py --obj_dir "$OBJ_DIR" --output_dir "$OUTPUT_DIR" --device "$DEVICE" --num_points "$NUM_POINTS"
        ;;
    "compare")
        if [ -z "$OBJ_FILE" ] || [ -z "$TEXT" ]; then
            echo -e "${RED}Error: Both --obj-file and --text are required${NC}"
            exit 1
        fi
        check_model || exit 1
        echo -e "${YELLOW}Comparing '$OBJ_FILE' with '$TEXT'${NC}"
        python ulip2_encoder.py --obj_file "$OBJ_FILE" --text "$TEXT" --compare --device "$DEVICE" --num_points "$NUM_POINTS"
        ;;
    "build-db")
        if [ -z "$OBJ_DIR" ]; then
            echo -e "${RED}Error: --obj-dir is required${NC}"
            exit 1
        fi
        check_model || exit 1
        echo -e "${YELLOW}Building vector database from: $OBJ_DIR${NC}"
        python ulip2_encoder.py --obj_dir "$OBJ_DIR" --build_db --db_path "$DB_PATH" --device "$DEVICE" --num_points "$NUM_POINTS"
        ;;
    "search-db")
        if [ -z "$TEXT" ] && [ -z "$OBJ_FILE" ]; then
            echo -e "${RED}Error: Either --text or --obj-file is required for search${NC}"
            exit 1
        fi
        check_model || exit 1
        if [ ! -z "$TEXT" ]; then
            echo -e "${YELLOW}Searching database for text: $TEXT${NC}"
            python ulip2_encoder.py --search_db --db_path "$DB_PATH" --text "$TEXT" --top_k "$TOP_K" --device "$DEVICE"
        else
            echo -e "${YELLOW}Searching database for model: $OBJ_FILE${NC}"
            python ulip2_encoder.py --search_db --db_path "$DB_PATH" --obj_file "$OBJ_FILE" --top_k "$TOP_K" --device "$DEVICE" --num_points "$NUM_POINTS"
        fi
        ;;
    "")
        echo -e "${RED}Error: No command specified${NC}"
        print_help
        exit 1
        ;;
    *)
        echo -e "${RED}Error: Unknown command: $COMMAND${NC}"
        print_help
        exit 1
        ;;
esac
