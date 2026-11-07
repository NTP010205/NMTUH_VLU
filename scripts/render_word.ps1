param(
    [string]$InputDocx = 'report/Bao_cao_NMTUH.docx',
    [string]$OutputDirectory = 'report/qa/final'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$inputPath = if ([IO.Path]::IsPathRooted($InputDocx)) { $InputDocx } else { Join-Path $projectRoot $InputDocx }
$outputPath = if ([IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory } else { Join-Path $projectRoot $OutputDirectory }
$inputPath = (Resolve-Path -LiteralPath $inputPath).Path
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
$outputPath = (Resolve-Path -LiteralPath $outputPath).Path
$pdfPath = Join-Path $outputPath ([IO.Path]::GetFileNameWithoutExtension($inputPath) + '.pdf')
$wordInstance = $null
$document = $null
$exportComplete = $false
try {
    $wordInstance = New-Object -ComObject Word.Application
    $wordInstance.Visible = $false
    $wordInstance.DisplayAlerts = 0
    $document = $wordInstance.Documents.Open($inputPath, $false, $true)
    $document.Fields.Update() | Out-Null
    $document.Repaginate()
    $pageCount = $document.ComputeStatistics(2)
    $document.ExportAsFixedFormat($pdfPath, 17)
    $exportComplete = Test-Path -LiteralPath $pdfPath
    @{ renderer = 'Microsoft Word native PDF export'; input = $inputPath; pdf = $pdfPath; pages = $pageCount; visual_qa = 'pending' } |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $outputPath 'render.json') -Encoding utf8
    Write-Output "Rendered $pageCount pages to $pdfPath; visual inspection still required."
} finally {
    if ($null -ne $document) { $document.Close(0); [Runtime.InteropServices.Marshal]::ReleaseComObject($document) | Out-Null }
    if ($null -ne $wordInstance) {
        try { $wordInstance.Quit() }
        catch {
            # Some Word installations exit when their last automation document
            # closes. The PDF already exists; an unavailable server then means
            # there is no live instance left to quit.
            if (-not ($exportComplete -and $_.Exception.Message -match '0x800706BA')) { throw }
        }
        [Runtime.InteropServices.Marshal]::ReleaseComObject($wordInstance) | Out-Null
    }
}
