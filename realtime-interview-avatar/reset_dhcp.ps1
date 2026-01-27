# ============================================
# DHCP로 복원 스크립트 (고정 IP 해제)
# 관리자 권한으로 실행 필요
# ============================================

$adapter = Get-NetAdapter | Where-Object { $_.Status -eq "Up" -and $_.Name -like "*Wi-Fi*" }

if (-not $adapter) {
    Write-Host "Wi-Fi 어댑터를 찾을 수 없습니다." -ForegroundColor Red
    exit 1
}

Write-Host "어댑터: $($adapter.Name)" -ForegroundColor Green
Write-Host "DHCP로 복원 중..." -ForegroundColor Cyan

try {
    # DHCP로 복원
    Set-NetIPInterface -InterfaceAlias $adapter.Name -Dhcp Enabled
    Set-DnsClientServerAddress -InterfaceAlias $adapter.Name -ResetServerAddresses

    Write-Host "DHCP 복원 완료!" -ForegroundColor Green

    # IP 갱신
    ipconfig /release
    ipconfig /renew

    Write-Host ""
    Get-NetIPAddress -InterfaceAlias $adapter.Name | Where-Object { $_.AddressFamily -eq "IPv4" }

} catch {
    Write-Host "오류 발생: $_" -ForegroundColor Red
}

Read-Host "Enter 키를 눌러 종료"
