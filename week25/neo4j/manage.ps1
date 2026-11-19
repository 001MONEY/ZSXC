param(
    [Parameter(Position = 0)]
    [ValidateSet('start', 'stop', 'restart', 'status', 'logs', 'pull', 'reset', 'shell', 'browser')]
    [string]$Action = 'status'
)

$ErrorActionPreference = 'Stop'
$ComposeFile = Join-Path $PSScriptRoot 'docker-compose.yml'

switch ($Action) {
    'start' {
        docker compose -f $ComposeFile up -d
        Write-Host 'Neo4j Browser : http://localhost:7474'
        Write-Host 'Bolt endpoint : bolt://localhost:7687  (neo4j / neo4j123456)'
    }
    'stop' {
        docker compose -f $ComposeFile stop
    }
    'restart' {
        docker compose -f $ComposeFile restart
    }
    'status' {
        docker compose -f $ComposeFile ps
    }
    'logs' {
        docker compose -f $ComposeFile logs --tail 100 neo4j
    }
    'pull' {
        docker compose -f $ComposeFile pull
    }
    'reset' {
        Write-Host 'This deletes all Neo4j data under volumes\neo4j ...' -ForegroundColor Yellow
        $answer = Read-Host 'Type YES to continue'
        if ($answer -ne 'YES') { Write-Host 'aborted'; exit 1 }
        docker compose -f $ComposeFile down -v
        Remove-Item -Recurse -Force (Join-Path $PSScriptRoot 'volumes\neo4j') -ErrorAction SilentlyContinue
        Write-Host 'data removed'
    }
    'shell' {
        docker exec -it neo4j cypher-shell -u neo4j -p neo4j123456
    }
    'browser' {
        Start-Process 'http://localhost:7474'
    }
}
