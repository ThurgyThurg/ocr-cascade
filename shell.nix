{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    ollama
    python313
    tesseract
    libGL
    zlib
    glib
    stdenv.cc.cc.lib
    libxcb              # opencv-python-headless still dlopens libxcb.so.1
    libx11              # …and libX11 / libXext on some cv2 wheels
    libxext
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
      pkgs.libGL
      pkgs.zlib
      pkgs.glib
      pkgs.stdenv.cc.cc.lib
      pkgs.libxcb
      pkgs.libx11
      pkgs.libxext
    ]}:/run/opengl-driver/lib:$LD_LIBRARY_PATH

    if [ -f venv/bin/activate ]; then
      source venv/bin/activate
    fi

    echo "OCR pipeline environment ready."
    echo "Run ./setup.sh to install Python deps and pull glm-ocr model."
    echo "Then: python ocr_pipeline.py <file.pdf> --engine all --compare"
  '';
}
