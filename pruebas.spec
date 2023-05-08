# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['pruebas.py'],
    pathex=[],
    binaries=[],
    datas=[('C:/Users/Admin/Desktop/face-recognition-with-liveness-web-login-master/face_recognition_and_liveness/face_liveness_detection/face_recognition2/', 'face_recognition2/'), ('C:/Users/Admin/anaconda3/Lib/site-packages/tensorflow/', 'tensorflow/'), ('C:/Users/Admin/anaconda3/Lib/site-packages/tensorflow/python/', 'tensorflow/python/'), ('C:/Users/Admin/anaconda3/Lib/site-packages/torch/', 'torch/'), ('C:/Users/Admin/Desktop/face-recognition-with-liveness-web-login-master/face_recognition_and_liveness/face_liveness_detection/src/', 'src/'), ('no_registrado.png', '.'), ('cuadro_facer.png', '.'), ('foto_falsa.png', '.'), ('label_encoder.pickle', '.'), ('liveness.model', '.'), ('C:/Users/Admin/Desktop/face-recognition-with-liveness-web-login-master/face_recognition_and_liveness/face_liveness_detection/face_detector/', 'face_detector/'), ('C:/Users/Admin/Desktop/face-recognition-with-liveness-web-login-master/face_recognition_and_liveness/face_liveness_detection/face_recognition_models/*', 'face_recognition_models/'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/util/_pywrap_utils.pyd', '.'), ('C:/Users/Admin/anaconda3/Lib/site-packages/torch/_C.cp39-win_amd64.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/util/_pywrap_nest.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/framework/_pywrap_python_api_dispatcher.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/framework/_op_def_library_pybind.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/framework/_op_def_registry.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/saved_model/pywrap_saved_model.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/lib/io/_pywrap_file_io.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/lib/io/_pywrap_record_io.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/util/_pywrap_tensor_float_32_execution.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/util/_pywrap_determinism.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/profiler/internal/_pywrap_traceme.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/framework/_proto_comparators.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/profiler/internal/_pywrap_profiler.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/util/_pywrap_tfprof.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/lib/core/_pywrap_py_func.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/lib/core/_pywrap_float8.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/lib/core/_pywrap_custom_casts.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/lib/core/_pywrap_bfloat16.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/data/experimental/service/_pywrap_server_lib.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/grappler/_pywrap_tf_cluster.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/grappler/_pywrap_tf_optimizer.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/client/_pywrap_events_writer.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/util/_pywrap_checkpoint_reader.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/client/_pywrap_device_lib.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/framework/_dtypes.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/framework/_test_metrics_util.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/platform/_pywrap_stacktrace_handler.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/platform/_pywrap_cpu_feature_guard.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/util/_pywrap_kernel_registry.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/framework/_pywrap_python_op_gen.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/client/_pywrap_tf_session.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/client/_pywrap_debug_events_writer.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/util/_pywrap_transform_graph.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/util/_pywrap_stat_summarizer.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/grappler/_pywrap_tf_item.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/framework/_pywrap_python_api_parameter_converter.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/framework/_pywrap_python_api_info.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/framework/_op_def_util.pyd', '.'), ('C:/Users/Admin/anaconda3/lib/site-packages/tensorflow/python/framework/_errors_test_helper.pyd', '.')],
    hiddenimports=['paho.mqtt.client', 'PIL', 'tensorflow', 'face_recognition', 'tkinter', 'numpy', 'warnings', 'sklearn', 'sklearn.neighbors._typedefs', 'imutils', 'pickle', 'cv2', 'copy', 'os', 'src.anti_spoof_predict', 'src.generate_patches', 'src.utility'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['googleapiclient.discovery'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='pruebas',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='pruebas',
)
