`gvardump` is used to dump global variables of a living C program without interrupting it.

## Advantage
* This tool directly reads the process memory to get the required data without interrupting the target process.
  In contrast, printing variables with gdb will interrupt it by attaching to the target process. 

* This tool supports basic C rvalue expressions.
  It can print arrays, structures, and nested member variables.

* This tool prints arrays and structs in a human readable format instead of cramming them on a single line.

## Dependence
gvardump currently uses gdb to resolve symbol addresses and type information, so gdb needs to be installed on the system. 

gvardump will not attach to the target process when calling gdb, and will not cause the process to be interrupted, so you can use it with confidence. 

gvardump has no other dependencies except gdb and python built-in libraries.

## Example
```shell script
root@ubuntu:/home/u/trace_test# ./gvardump.py 53670 -a 1 '*g_ss[0].sss[0].ps'
*((g_ss[0]).sss[0]).ps = {
    .a = 6,
    .sss = {
        {
            .bbb = 0,
            .ps = 0x563ca42a2020,
            .bs = {
                .m = 0,
            },
        },
        // other 9 elements are omit
    },
    .b = {
        {
            "",
            // other 331 elements are omit
        },
        // other 15 elements are omit
    },
    .g = {
        .grr = "<binary>" /* hex: a319b1000000 */,
    },
};
```

## Usage
```shell script
usage: gvardump.py [-h] [-v] [-x] [-e ELF_PATH] [-a ARRAY_MAX] [-s STRING_MAX]
                   pid expression

dump global variables of a living process without interrupting it

positional arguments:
  pid                   target process ID
  expression            a C rvalue expression

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         show debug information
  -x, --hex-string      dump byte array in hex instead of string
  -e ELF_PATH, --elf-path ELF_PATH
                        elf path to read symbols, `readlink /proc/$pid/exe` by
                        default
  -a ARRAY_MAX, --array-max ARRAY_MAX
                        maximum number of elements to display in an array
  -s STRING_MAX, --string-max STRING_MAX
                        displayed maximum string length

examples:
    * type of g_var1 is 'struct foo**', dump the second struct:
        ./gvardump.py 32311 '*g_var1[2]'
    * type of g_var2 is 'struct foo*', dump the first 5 elements:
        ./gvardump.py 32311 '(struct foo[5])*g_var2'
    * g_var3 points to a nested struct, dump the member:
        ./gvardump.py 32311 'g_var3->val.data'
    * there is a 'struct foo' at address 0x556159b32020, dump it:
        ./gvardump.py 32311 '(struct foo)0x556159b32020'
```

## Limitation
* Bitfields are not supported yet.

* Anonymous structures and unions are not supported yet.
