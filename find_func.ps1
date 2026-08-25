$out = Join-Path $env:TEMP 'llama_unwind.txt'
$lines = [System.IO.File]::ReadAllLines($out)
$target = 0x180072198
for ($i = 0; $i -lt $lines.Length; $i++) {
    if ($lines[$i] -match 'StartAddress: \(0x([0-9A-F]+)\)') {
        $start = [Convert]::ToInt64($matches[1], 16)
        if ($start -le $target) {
            # peek EndAddress on a following line
            for ($j = $i + 1; $j -lt [Math]::Min($i + 8, $lines.Length); $j++) {
                if ($lines[$j] -match 'EndAddress: \(0x([0-9A-F]+)\)') {
                    $end = [Convert]::ToInt64($matches[1], 16)
                    if ($end -gt $target) {
                        Write-Output ("FOUND: start=0x{0:X} end=0x{1:X} size=0x{2:X}" -f $start, $end, ($end-$start))
                        exit 0
                    }
                    break
                }
            }
        } else {
            Write-Output ("passed target without containment; first start > target = 0x{0:X}" -f $start)
            exit 1
        }
    }
}



