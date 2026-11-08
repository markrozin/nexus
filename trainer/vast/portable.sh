# Sourced by the scripts that run on Mark's own machine (prepare.sh, rent.sh,
# reap.sh, guardtest.sh) so they work the same under Git Bash on Windows, on
# Linux, and on macOS. run.sh runs on the Linux rental and never sources this.
#
# What differs, and how it is settled here:
#
#   - macOS ships BSD tools and lacks GNU `timeout`, `shuf` and `sha256sum`;
#     its `stat` takes different flags. macos-bin/ holds small stand-ins for
#     exactly the forms these scripts use, put first on PATH on macOS only.
#     guardtest.sh needs them too, since it exercises run.sh locally.
#   - Built binaries end in .exe on Windows only: use "datacheck$EXE".
#   - Python is `python` on Windows and `python3` on macOS. PYTHON overrides.
#   - The vastai CLI lives in a Windows user directory there and on PATH
#     elsewhere. VASTAI overrides.

case "$(uname -s)" in
    Darwin)
        PORTABLE_OS=macos
        PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/macos-bin:$PATH"
        export PATH
        # bsdtar would otherwise add AppleDouble ._ files and extended
        # attributes to the bundle, which tar on the rental warns about.
        export COPYFILE_DISABLE=1
        ;;
    MINGW*|MSYS*|CYGWIN*) PORTABLE_OS=windows ;;
    *) PORTABLE_OS=linux ;;
esac

if [ "$PORTABLE_OS" = windows ]; then EXE=.exe; else EXE=; fi

if [ -z "${PYTHON:-}" ]; then
    if command -v python3 > /dev/null 2>&1 && python3 -c '' 2> /dev/null; then
        PYTHON=python3
    else
        PYTHON=python
    fi
fi

if [ -z "${VASTAI:-}" ]; then
    if [ "$PORTABLE_OS" = windows ]; then
        VASTAI=/c/Users/Mark/AppData/Roaming/Python/Python314/Scripts/vastai.exe
    else
        VASTAI="$(command -v vastai 2> /dev/null || echo vastai)"
    fi
fi

# Seconds since the epoch -> HH:MM:SS local time. GNU date takes -d @N, BSD -r N.
clock_time() {
    date -d "@$1" +%T 2> /dev/null || date -r "$1" +%T
}

# In-place sed that both GNU and BSD accept: BSD requires a backup suffix
# argument, GNU requires it attached, so give one and delete the backup.
sed_in_place() {
    local file="${!#}"
    sed -i.portable-bak "$@" && rm -f "$file.portable-bak"
}
