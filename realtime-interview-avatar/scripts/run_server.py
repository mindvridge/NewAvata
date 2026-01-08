#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
간단한 서버 실행 스크립트
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    # app.py에서 직접 app 가져오기 (src.server.__init__ 우회)
    import importlib.util
    app_path = project_root / "src" / "server" / "app.py"
    spec = importlib.util.spec_from_file_location("app_module", app_path)
    app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_module)
    app = app_module.app

    # 루트 경로에서 index.html 서빙
    static_dir = project_root / "src" / "server" / "static"

    @app.get("/")
    async def serve_index():
        """메인 페이지 서빙"""
        index_path = static_dir / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return {"message": "Interview Avatar API", "docs": "/docs"}

    @app.get("/favicon.ico")
    async def favicon():
        return {"status": "ok"}

    @app.get("/styles.css")
    async def serve_styles():
        """CSS 파일 서빙"""
        css_path = static_dir / "styles.css"
        if css_path.exists():
            return FileResponse(str(css_path), media_type="text/css")
        return {"error": "styles.css not found"}

    @app.get("/main.js")
    async def serve_main_js():
        """JavaScript 파일 서빙"""
        js_path = static_dir / "main.js"
        if js_path.exists():
            return FileResponse(str(js_path), media_type="application/javascript")
        return {"error": "main.js not found"}

    print("\n" + "=" * 60)
    print("  Realtime Interview Avatar Server")
    print("=" * 60)
    print(f"\n  API:     http://localhost:8000/api/health")
    print(f"  Docs:    http://localhost:8000/docs")
    print(f"  WebApp:  http://localhost:8000/")
    print(f"\n  WebSocket: ws://localhost:8000/ws/realtime/{{session_id}}")
    print("\n" + "=" * 60 + "\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
