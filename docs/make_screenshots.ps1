# Regenerates the README screenshots with headless Microsoft Edge.
# Start the dashboard first (FermenTrack.bat), then run:  powershell -File docs\make_screenshots.ps1
$edge = @("C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
          "C:\Program Files\Microsoft\Edge\Application\msedge.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
$out = Join-Path $PSScriptRoot "screenshots"
New-Item -ItemType Directory -Force $out | Out-Null
$shots = @{
    "batch_monitor"     = @("page=monitor&batch=68&t=36", 2300)
    "probe_fouling"     = @("page=monitor&batch=62&t=40", 1500)
    "plant_overview"    = @("page=overview", 1500)
    "pls_calibration"   = @("page=performance&section=pls", 1700)
    "yield_forecast"    = @("page=performance&section=yield", 1800)
}
foreach ($name in $shots.Keys) {
    $q, $h = $shots[$name]
    # separate profile + -Wait so each screenshot finishes before the next starts
    $profile = Join-Path $env:TEMP "fermentrack-edge-profile"
    Start-Process -FilePath $edge -Wait -WindowStyle Hidden -ArgumentList @(
        "--headless=new", "--disable-gpu", "--hide-scrollbars", "--user-data-dir=$profile",
        "--window-size=1500,$h", "--virtual-time-budget=30000",
        "--screenshot=$out\$name.png", "http://localhost:8501/?$q")
}
Get-ChildItem $out
