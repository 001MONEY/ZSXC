# Milvus Standalone helper (ASCII only: Windows PowerShell 5.1 reads .ps1 as ANSI)
# Usage:
#   .\manage.ps1 start    # start etcd + minio + milvus-standalone
#   .\manage.ps1 stop     # stop and remove containers (data kept in .\volumes)
#   .\manage.ps1 status   # container status + healthz
#   .\manage.ps1 logs     # follow standalone logs
#   .\manage.ps1 webui    # open built-in WebUI http://127.0.0.1:9091/webui
#   .\manage.ps1 gui      # start Milvus + Attu console http://localhost:8000
#   .\manage.ps1 pull     # pre-pull images
#   .\manage.ps1 reset    # stop and delete .\volumes (DANGEROUS)
param(
    [ValidateSet('start', 'stop', 'status', 'logs', 'webui', 'gui', 'pull', 'reset')]
    [string]$Action = 'start'
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)
$compose = 'docker-compose.yml'
$composeAttu = 'docker-compose.attu.yml'

switch ($Action) {
    'start' {
        docker compose -f $compose up -d
        docker compose -f $compose ps
    }
    'stop' {
        docker compose -f $compose down
    }
    'status' {
        docker compose -f $compose ps
        try {
            $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 http://127.0.0.1:9091/healthz
            "healthz => $($r.StatusCode) $($r.Content)"
        }
        catch {
            "healthz unavailable: $($_.Exception.Message)"
        }
    }
    'logs' {
        docker compose -f $compose logs -f --tail 100 standalone
    }
    'webui' {
        Start-Process 'http://127.0.0.1:9091/webui'
    }
    'gui' {
        docker compose -f $compose -f $composeAttu up -d
        docker compose -f $compose -f $composeAttu ps
        Start-Process 'http://localhost:8000'
    }
    'pull' {
        docker compose -f $compose pull
    }
    'reset' {
        docker compose -f $compose down
        Remove-Item -Recurse -Force .\volumes -ErrorAction SilentlyContinue
        'volumes removed; next start creates a fresh instance'
    }
}
