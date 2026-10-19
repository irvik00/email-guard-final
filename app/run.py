"""Single-process launcher; loopback locally, PORT/host configured by the operator."""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', '8000')))
    parser.add_argument('--host', choices=['127.0.0.1', '0.0.0.0'], default=os.environ.get('EMAIL_GUARD_HOST', '127.0.0.1'),
                        help='0.0.0.0 внутри контейнера; публичный адрес задаётся оператором')
    parser.add_argument('--disable-laya', action='store_true', help='Оставить доступным только baseline')
    args = parser.parse_args()
    if args.disable_laya:
        os.environ['EMAIL_GUARD_DISABLE_LAYA'] = '1'
    import uvicorn
    uvicorn.run('app.web:app', host=args.host, port=args.port, workers=1, reload=False,
                access_log=False, log_level='warning', timeout_keep_alive=5)


if __name__ == '__main__':
    main()
