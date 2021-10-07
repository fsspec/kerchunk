import base64
import ujson as json


def format_for_output(mapping):
    out = {}
    for k, v in mapping.items():
        if isinstance(v, bytes):
            try:
                # easiest way to test if data is ascii
                out[k] = v.decode('ascii')
                try:
                    # minify json
                    out[k] = json.dumps(json.loads(out[k]))
                except:
                    pass
            except UnicodeDecodeError:
                out[k] = (b"base64:" + base64.b64encode(v)).decode()
        else:
            out[k] = v
    return {"version": 1, "refs": out}
