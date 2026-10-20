"""FastAPI API; configured local/public host, streaming uploads never touch disk."""
import asyncio
import os
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from .contract import MAX_BYTES, MODES
from .inference import Engine, InputError
from .deployment_config import deployment_settings

STATIC = Path(__file__).parent / 'static'
ERROR_TEXT = {'empty_file': 'Файл пуст.', 'invalid_eml': 'Не удалось прочитать письмо.',
              'file_too_large': 'Размер письма превышает 10 MiB.',
              'invalid_request': 'Неверные параметры запроса.',
              'busy': 'Приложение занято. Повторите запрос позже.',
              'unsupported_media_type': 'Передайте исходные байты .eml с типом message/rfc822.',
              'origin_forbidden': 'Запрос разрешён только из интерфейса этого сервиса.',
              'internal_error': 'Обработка не выполнена. Вердикт отсутствует.'}


class Mode(str, Enum):
    baseline = 'baseline'
    recall = 'recall'
    review = 'review'


def error(code, status):
    return JSONResponse({'status': 'error', 'decision': None, 'error': code, 'message': ERROR_TEXT[code]}, status_code=status)


def create_app(engine=None, enable_laya=True):
    @asynccontextmanager
    async def lifespan(application):
        application.state.engine = engine or await asyncio.to_thread(Engine, enable_laya)
        application.state.slots = asyncio.Semaphore(2)
        try:
            yield
        finally:
            if engine is None:
                await asyncio.to_thread(application.state.engine.close)

    application = FastAPI(title='Email Guard — анализ .eml', version='1.0.0', lifespan=lifespan,
                          docs_url=None, redoc_url=None, description='Три фиксированных режима. Никакого обучения и загрузки моделей через API.')
    allowed_hosts, public_origins = deployment_settings()
    application.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    @application.middleware('http')
    async def security(request: Request, call_next):
        origin = request.headers.get('origin')
        if request.method == 'POST' and origin and origin not in public_origins and origin != str(request.base_url).rstrip('/'):
            response = error('origin_forbidden', 403)
        else:
            response = await call_next(request)
        response.headers.update({'Content-Security-Policy': "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
                                 'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'no-referrer', 'Cache-Control': 'no-store'})
        return response

    @application.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return error('invalid_request', 422)

    @application.exception_handler(Exception)
    async def unexpected_error(request, exc):
        return error('internal_error', 500)

    @application.get('/', include_in_schema=False)
    def index():
        return FileResponse(STATIC / 'index.html', media_type='text/html')

    @application.get('/evaluation', include_in_schema=False)
    def evaluation():
        return FileResponse(STATIC / 'evaluation.html', media_type='text/html')

    @application.get('/evaluation-data.json', include_in_schema=False)
    def evaluation_data():
        return FileResponse(STATIC / 'evaluation.json', media_type='application/json')

    @application.get('/assets/app.js', include_in_schema=False)
    def javascript():
        return FileResponse(STATIC / 'app.js', media_type='text/javascript')

    @application.get('/assets/style.css', include_in_schema=False)
    def stylesheet():
        return FileResponse(STATIC / 'style.css', media_type='text/css')

    @application.get('/api-docs', include_in_schema=False)
    def docs():
        return FileResponse(STATIC / 'api.html', media_type='text/html')

    @application.get('/health', summary='Жив ли процесс приложения')
    def health():
        return {'status': 'alive', 'upload_persistence': False}

    @application.get('/ready', summary='Готовность компонентов и каждого режима')
    def ready(request: Request, mode: Mode = Mode.baseline):
        result = request.app.state.engine.readiness()
        return JSONResponse(result, status_code=200 if result['modes'][mode.value] else 503)

    @application.post('/api/analyze', summary='Проанализировать одно письмо из байтов .eml',
        description='Тело запроса — raw .eml, без multipart и без имени файла. Потоковый лимит 10 MiB. 200: решение/отказ из-за признаков/needs_review; 503: необходимый компонент недоступен; 413: лимит; 400: пустой/повреждённый файл; 429: занято. Сведения письма не сохраняются.',
        openapi_extra={'requestBody': {'required': True, 'content': {'message/rfc822': {'schema': {'type': 'string', 'format': 'binary'}}}}},
        responses={400: {'description': 'Некорректное письмо'}, 413: {'description': 'Больше 10 MiB'},
                   415: {'description': 'Неверный Content-Type'}, 429: {'description': 'Занято'}, 503: {'description': 'Компонент недоступен'}})
    async def analyze(request: Request, mode: Mode = Mode.baseline):
        if set(request.query_params) - {'mode'}:
            return error('invalid_request', 422)
        if request.headers.get('content-type', '').split(';')[0].strip().lower() not in ('message/rfc822', 'application/octet-stream'):
            return error('unsupported_media_type', 415)
        try:
            if int(request.headers.get('content-length', '0')) > MAX_BYTES:
                return error('file_too_large', 413)
        except ValueError:
            return error('invalid_request', 400)
        # Bound simultaneous parsing, byte buffers and TF-IDF, in addition to
        # the separate one-request lock of the single persistent neural worker.
        try:
            await asyncio.wait_for(request.app.state.slots.acquire(), timeout=0.1)
        except TimeoutError:
            return error('busy', 429)
        try:
            content = bytearray()
            async for chunk in request.stream():
                if len(content) + len(chunk) > MAX_BYTES:
                    return error('file_too_large', 413)
                content.extend(chunk)
            result = await asyncio.to_thread(request.app.state.engine.analyze, bytes(content), mode.value)
            return JSONResponse(result, status_code=503 if result['infrastructure_failure'] else 200)
        except InputError as exc:
            code = str(exc)
            return error(code, 413 if code == 'file_too_large' else 400)
        except Exception:
            # No raw exception logging: library exceptions may contain email data.
            return error('internal_error', 500)
        finally:
            request.app.state.slots.release()

    return application


app = create_app(enable_laya=os.environ.get('EMAIL_GUARD_DISABLE_LAYA') != '1')
