#!/bin/bash

# clear_python_cache.sh - Clear all Python cache files in a directory

# Function to display usage
usage() {
    echo "Usage: $0 [directory]"
    echo "  directory: Path to directory to clear cache from (default: current directory)"
    echo ""
    echo "Examples:"
    echo "  $0                    # Clear cache in current directory"
    echo "  $0 /path/to/project   # Clear cache in specific directory"
    echo "  $0 .                  # Clear cache in current directory"
}

# Function to clear Python cache
clear_python_cache() {
    local target_dir="$1"
    
    echo "Clearing Python cache in: $target_dir"
    echo "=================================="
    
    # Count files before deletion
    local __pycache_count=$(find "$target_dir" -type d -name "__pycache__" 2>/dev/null | wc -l)
    local pyc_count=$(find "$target_dir" -name "*.pyc" 2>/dev/null | wc -l)
    local pyo_count=$(find "$target_dir" -name "*.pyo" 2>/dev/null | wc -l)
    
    echo "Found:"
    echo "  - $__pycache_count __pycache__ directories"
    echo "  - $pyc_count .pyc files"
    echo "  - $pyo_count .pyo files"
    echo ""
    
    if [ $((__pycache_count + pyc_count + pyo_count)) -eq 0 ]; then
        echo "No Python cache files found."
        return 0
    fi
    
    # Remove __pycache__ directories
    if [ $__pycache_count -gt 0 ]; then
        echo "Removing __pycache__ directories..."
        find "$target_dir" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
        echo "  ✓ Removed $__pycache_count __pycache__ directories"
    fi
    
    # Remove .pyc files
    if [ $pyc_count -gt 0 ]; then
        echo "Removing .pyc files..."
        find "$target_dir" -name "*.pyc" -delete 2>/dev/null
        echo "  ✓ Removed $pyc_count .pyc files"
    fi
    
    # Remove .pyo files
    if [ $pyo_count -gt 0 ]; then
        echo "Removing .pyo files..."
        find "$target_dir" -name "*.pyo" -delete 2>/dev/null
        echo "  ✓ Removed $pyo_count .pyo files"
    fi
    
    echo ""
    echo "✅ Python cache cleared successfully!"
}

# Main script
main() {
    # Get target directory (default to current directory)
    local target_dir="${1:-.}"
    
    # Check if directory exists
    if [ ! -d "$target_dir" ]; then
        echo "Error: Directory '$target_dir' does not exist."
        exit 1
    fi
    
    # Convert to absolute path
    target_dir=$(realpath "$target_dir")
    
    # Confirm before proceeding
    echo "This will clear all Python cache files in: $target_dir"
    echo "This includes:"
    echo "  - All __pycache__ directories"
    echo "  - All .pyc files"
    echo "  - All .pyo files"
    echo ""
    read -p "Do you want to continue? (y/N): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        clear_python_cache "$target_dir"
    else
        echo "Operation cancelled."
        exit 0
    fi
}

# Handle help flags
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

# Run main function
main "$@"