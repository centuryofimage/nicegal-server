# Embedded in the Rust binary; inputs are environment variables, not executable text.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$partial = $null
try {
    $endpoint = $env:HF_ENDPOINT
    if (!$endpoint) { $endpoint = 'https://huggingface.co' }
    $endpoint = $endpoint.TrimEnd('/')
    $repo = ($env:NICEGAL_HF_REPO.Split('/') | ForEach-Object { [Uri]::EscapeDataString($_) }) -join '/'
    $revision = [Uri]::EscapeDataString($env:NICEGAL_HF_REVISION)
    $headers = @{}
    if ($env:NICEGAL_HF_TOKEN) { $headers.Authorization = 'Bearer ' + $env:NICEGAL_HF_TOKEN }
    $metadata = Invoke-RestMethod -UseBasicParsing -Uri "$endpoint/api/models/$repo/revision/${revision}?blobs=true" -Headers $headers -TimeoutSec 60
    $commit = [string]$metadata.sha
    if ($commit -notmatch '^[a-f0-9]{40}$') { throw 'Invalid Hub commit identifier.' }
    $file = @($metadata.siblings | Where-Object { $_.rfilename -ceq $env:NICEGAL_HF_FILE })
    if ($file.Count -ne 1) { throw 'Requested file is missing from Hub metadata.' }
    $file = $file[0]
    if ($null -eq $file.size -or [long]$file.size -le 0) { throw 'Missing or invalid Hub file size.' }
    $folder = 'models--' + $env:NICEGAL_HF_REPO.Replace('/', '--')
    $root = Join-Path $env:NICEGAL_HF_CACHE $folder
    $target = Join-Path (Join-Path $root "snapshots/$commit") $env:NICEGAL_HF_FILE
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($target)) | Out-Null
    $partial = "$target.$([guid]::NewGuid().ToString('N')).partial"
    $filename = ($env:NICEGAL_HF_FILE.Split('/') | ForEach-Object { [Uri]::EscapeDataString($_) }) -join '/'
    [Console]::Out.WriteLine("PROGRESS 0 $($file.size)")
    $request = [Net.HttpWebRequest]::Create("$endpoint/$repo/resolve/$commit/$filename")
    $request.Timeout = 1800000
    $request.ReadWriteTimeout = 1800000
    foreach ($key in $headers.Keys) { $request.Headers[$key] = $headers[$key] }
    $response = $request.GetResponse()
    try {
        $inputStream = $response.GetResponseStream()
        $outputStream = [IO.File]::Create($partial)
        try {
            $buffer = New-Object byte[] 65536
            $downloaded = [long]0
            $timer = [Diagnostics.Stopwatch]::StartNew()
            while (($count = $inputStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
                $outputStream.Write($buffer, 0, $count)
                $downloaded += $count
                if ($timer.ElapsedMilliseconds -ge 100) {
                    [Console]::Out.WriteLine("PROGRESS $downloaded $($file.size)")
                    $timer.Restart()
                }
            }
            [Console]::Out.WriteLine("PROGRESS $downloaded $($file.size)")
        } finally { $outputStream.Dispose(); $inputStream.Dispose() }
    } finally { $response.Dispose() }
    if ((Get-Item -LiteralPath $partial).Length -ne [long]$file.size) { throw 'Downloaded file size does not match Hub metadata.' }
    if ($file.lfs.sha256) {
        $hash = [Security.Cryptography.SHA256]::Create()
        $stream = [IO.File]::OpenRead($partial)
        try {
            if ([BitConverter]::ToString($hash.ComputeHash($stream)).Replace('-', '') -ine $file.lfs.sha256) { throw 'Downloaded file SHA-256 mismatch.' }
        } finally { $stream.Dispose(); $hash.Dispose() }
    } else {
        if ($file.blobId -notmatch '^[a-f0-9]{40}$') { throw 'Missing Git blob hash.' }
        $hash = [Security.Cryptography.SHA1]::Create()
        $stream = [IO.File]::OpenRead($partial)
        try {
            $prefix = [Text.Encoding]::UTF8.GetBytes("blob $($file.size)" + [char]0)
            $hash.TransformBlock($prefix, 0, $prefix.Length, $prefix, 0) | Out-Null
            $buffer = New-Object byte[] 65536
            while (($count = $stream.Read($buffer, 0, $buffer.Length)) -gt 0) {
                $hash.TransformBlock($buffer, 0, $count, $buffer, 0) | Out-Null
            }
            $hash.TransformFinalBlock(@(), 0, 0) | Out-Null
            if ([BitConverter]::ToString($hash.Hash).Replace('-', '') -ine $file.blobId) { throw 'Downloaded Git blob hash mismatch.' }
        } finally { $stream.Dispose(); $hash.Dispose() }
    }
    # Move only verified complete files into snapshots. Publish refs last.
    Move-Item -LiteralPath $partial -Destination $target -Force
    $partial = $null
    $ref = Join-Path (Join-Path $root 'refs') $env:NICEGAL_HF_REVISION
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($ref)) | Out-Null
    $partial = "$ref.$([guid]::NewGuid().ToString('N')).partial"
    [IO.File]::WriteAllText($partial, $commit, [Text.Encoding]::ASCII)
    Move-Item -LiteralPath $partial -Destination $ref -Force
    $partial = $null
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
} finally {
    if ($partial -and (Test-Path -LiteralPath $partial)) { Remove-Item -LiteralPath $partial -Force }
}
