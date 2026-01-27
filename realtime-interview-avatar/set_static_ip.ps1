# ============================================
# Windows Wi-Fi 고정 IP 설정 스크립트
# 관리자 권한으로 실행 필요
# ============================================

# 현재 Wi-Fi 어댑터 이름 확인
$adapter = Get-NetAdapter | Where-Object { $_.Status -eq "Up" -and $_.Name -like "*Wi-Fi*" }

if (-not $adapter) {
    Write-Host "Wi-Fi 어댑터를 찾을 수 없습니다." -ForegroundColor Red
    exit 1
}

Write-Host "어댑터 발견: $($adapter.Name)" -ForegroundColor Green

# 고정 IP 설정값 (필요시 수정)
$IPAddress = "172.16.10.45"
$PrefixLength = 24  # 255.255.255.0
$Gateway = "172.16.10.1"
$DNS = @("8.8.8.8", "8.8.4.4")  # Google DNS (공유오피스 DNS로 변경 가능)

Write-Host ""
Write-Host "설정할 고정 IP 정보:" -ForegroundColor Cyan
Write-Host "  IP 주소: $IPAddress"
Write-Host "  서브넷: /$PrefixLength (255.255.255.0)"
Write-Host "  게이트웨이: $Gateway"
Write-Host "  DNS: $($DNS -join ', ')"
Write-Host ""

$confirm = Read-Host "이 설정으로 진행하시겠습니까? (Y/N)"
if ($confirm -ne "Y" -and $confirm -ne "y") {
    Write-Host "취소되었습니다." -ForegroundColor Yellow
    exit 0
}

try {
    # 기존 IP 설정 제거
    Remove-NetIPAddress -InterfaceAlias $adapter.Name -Confirm:$false -ErrorAction SilentlyContinue
    Remove-NetRoute -InterfaceAlias $adapter.Name -Confirm:$false -ErrorAction SilentlyContinue

    # 고정 IP 설정
    New-NetIPAddress -InterfaceAlias $adapter.Name -IPAddress $IPAddress -PrefixLength $PrefixLength -DefaultGateway $Gateway

    # DNS 설정
    Set-DnsClientServerAddress -InterfaceAlias $adapter.Name -ServerAddresses $DNS

    Write-Host ""
    Write-Host "고정 IP 설정 완료!" -ForegroundColor Green
    Write-Host ""

    # 결과 확인
    Get-NetIPAddress -InterfaceAlias $adapter.Name | Where-Object { $_.AddressFamily -eq "IPv4" }

} catch {
    Write-Host "오류 발생: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "다른 컴퓨터에서 접근 URL:" -ForegroundColor Cyan
Write-Host "  http://$IPAddress:5000" -ForegroundColor Yellow
Write-Host ""
Read-Host "Enter 키를 눌러 종료"
