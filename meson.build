# meson.build

project('thefinals', 'c',
  default_options: ['c_std=c11', 'warning_level=2'])

onnx_path = 'C:/Users/Security/Desktop/Security Files/Eshan VOR/new AIRPLANE defincesystem/onnxruntime-win-x64-gpu-1.23.1'
onnx_inc = include_directories(onnx_path + '/include')
onnx_lib_dir = onnx_path + '/lib'

cc = meson.get_compiler('c')

onnx_dep = declare_dependency(
  include_directories: onnx_inc,
  link_args: ['-L' + onnx_lib_dir, '-lonnxruntime']
)

win_deps = [
  cc.find_library('user32'),
  cc.find_library('gdi32'),
  cc.find_library('kernel32')
]

executable('thefinals',
  sources: ['thefinal.c'],
  dependencies: [onnx_dep, win_deps]
)