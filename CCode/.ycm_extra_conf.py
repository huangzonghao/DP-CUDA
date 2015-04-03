import os
import ycm_core

flags = [

# Warnings
'-Wall',
'-Wextra',
'-Werror',
# '-Wc++98-compat',
'-Wno-long-long',
'-Wno-variadic-macros',

# Extras
'-fexceptions',
'-DNDEBUG',

# Change this to '-std=c99' if you write C!
'-std=c++11',

# And 'c' here!
'-x',
'c++',

# System headers
'-isystem',
'/usr/include/c++/4.8.2',
'-isystem',
'/usr/include/c++/4.8.2/backward',
'-isystem',
'/usr/include/c++/4.8.2/x86_64-amazon-linux',
'-isystem',
'/usr/local/include',
'-isystem',
'/usr/include',

# Project headers
'-I',
'./src',

# Cuda
'-I',
'/usr/local/cuda/include',
'-I',
'/usr/local/cuda/samples/common/inc',

]


compilation_database_folder = ''

if compilation_database_folder:
  database = ycm_core.CompilationDatabase( compilation_database_folder )
else:
  database = None


def DirectoryOfThisScript():
  return os.path.dirname( os.path.abspath( __file__ ) )


def MakeRelativePathsInFlagsAbsolute( flags, working_directory ):
  if not working_directory:
    return list( flags )
  new_flags = []
  make_next_absolute = False
  path_flags = [ '-isystem', '-I', '-iquote', '--sysroot=' ]
  for flag in flags:
    new_flag = flag

    if make_next_absolute:
      make_next_absolute = False
      if not flag.startswith( '/' ):
        new_flag = os.path.join( working_directory, flag )

    for path_flag in path_flags:
      if flag == path_flag:
        make_next_absolute = True
        break

      if flag.startswith( path_flag ):
        path = flag[ len( path_flag ): ]
        new_flag = path_flag + os.path.join( working_directory, path )
        break

    if new_flag:
      new_flags.append( new_flag )
  return new_flags


def FlagsForFile( filename ):
  if database:
    # Bear in mind that compilation_info.compiler_flags_ does NOT return a
    # python list, but a "list-like" StringVec object
    compilation_info = database.GetCompilationInfoForFile( filename )
    final_flags = MakeRelativePathsInFlagsAbsolute(
      compilation_info.compiler_flags_,
      compilation_info.compiler_working_dir_ )
  else:
    relative_to = DirectoryOfThisScript()
    final_flags = MakeRelativePathsInFlagsAbsolute( flags, relative_to )

  return {
    'flags': final_flags,
    'do_cache': True
  }
