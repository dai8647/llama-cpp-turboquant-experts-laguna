param([string]$Prompt = 'こんにちは、自己紹介してください。', [int]$N = 100)
$body = @{ prompt = $Prompt; n_predict = $N; temperature = 0.4; cache_prompt = $true } | ConvertTo-Json
try {
    $r = Invoke-RestMethod -Uri 'http://127.0.0.1:8082/completion' -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 600
    $res = @{
        ok = $true
        prompt = $Prompt
        predicted = $r.tokens_predicted
        evaluated = $r.tokens_evaluated
        tps = [math]::Round($r.timings.predicted_per_second, 3)
        pps = [math]::Round($r.timings.prompt_per_second, 3)
        content = $r.content
    } | ConvertTo-Json -Depth 3
} catch {
    $res = @{ ok = $false; error = $_.Exception.Message } | ConvertTo-Json
}
Set-Content -Path 'C:\Users\dai86\llama-cpp-turboquant-experts-laguna\result.json' -Value $res -Encoding UTF8
