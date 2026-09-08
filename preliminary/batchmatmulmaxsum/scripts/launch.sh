#!/usr/bin/env bash
set -eo pipefail
# launch.sh CANN_ROOT executable case_dir B M N K dtype TX1 TX2 repeats [tiling_key [warmups]]
cann_root="$1"
executable="$2"
case_dir="$3"
shift 3
source "${cann_root}/set_env.sh"
cd "${case_dir}"
args=("$1" "$2" "$3" "$4" "$5" "$6" "$7" "${case_dir}" "$8")
if [[ $# -ge 9 ]]; then
    args+=("$9")
fi
if [[ $# -ge 10 ]]; then
    args+=("${10}")
fi
exec "${executable}" "${args[@]}"
