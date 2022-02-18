#!/usr/bin/env python
from __future__ import print_function
import os
import re
import sys
import codecs
import subprocess
import argparse
import select
import platform
import struct
import functools
import traceback

verbose = False


def log_arg_ret(func):
    @functools.wraps(func)
    def _func(*args, **kwargs):
        ret = 'Error'
        try:
            ret = func(*args, **kwargs)
        finally:
            if verbose:
                args = args[1:]
                arg_str1 = ', '.join([str(i) for i in args])
                arg_str2 = ', '.join([str(k) + '=' + str(v)
                                      for k, v in kwargs.items()])
                if arg_str1 and arg_str2:
                    arg_str = arg_str1 + ', ' + arg_str2
                else:
                    arg_str = arg_str1 + arg_str2
                print("call: %s(%s) = '%s'" % (func.__name__, arg_str, ret),
                      file=sys.stderr)
        return ret

    return _func


def log_exception():
    if not verbose:
        return
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_exception(exc_type, exc_value, exc_traceback,
                              file=sys.stderr)


class Token(object):
    pass


class AST(object):
    pass


class Symbol(Token, AST):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value


class Number(Token, AST):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str('0x%x' % self.value)


class Rarrow(Token):
    def __str__(self):
        return '->'


class Dot(Token):
    def __str__(self):
        return '.'


class Asterisk(Token):
    def __str__(self):
        return '*'


class Lparen(Token):
    def __str__(self):
        return '('


class Rparen(Token):
    def __str__(self):
        return ')'


class Lsquare(Token):
    def __str__(self):
        return '['


class Rsquare(Token):
    def __str__(self):
        return ']'


class Struct(Token):
    def __str__(self):
        return 'struct'


class Lexer(object):
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.pos_history = []

    def backward(self):
        self.pos = self.pos_history.pop()

    def next_token(self):
        old_pos = self.pos
        try:
            token = self._next_token()
        except:
            self.pos = old_pos
            raise
        else:
            self.pos_history.append(old_pos)
        return token

    def _next_token(self):
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

        if self.pos >= len(self.text):
            return None

        value = self.text[self.pos]
        if value == '(':
            self.pos += 1
            return Lparen()
        elif value == ')':
            self.pos += 1
            return Rparen()
        elif value == '.':
            self.pos += 1
            return Dot()
        elif value == '*':
            self.pos += 1
            return Asterisk()
        elif value == '[':
            self.pos += 1
            return Lsquare()
        elif value == ']':
            self.pos += 1
            return Rsquare()
        elif self.pos < len(self.text) - 1 and \
                self.text[self.pos: self.pos + 2] == '->':
            self.pos += 2
            return Rarrow()
        elif value.isdigit():
            num_str = ''
            while self.pos < len(self.text) and self.text[self.pos].isalnum():
                num_str += self.text[self.pos]
                self.pos += 1
            return Number(int(num_str, 0))
        elif value.isalpha() or value == '_':
            symbol_str = ''
            while self.pos < len(self.text) and \
                    (self.text[self.pos].isalnum() or
                     self.text[self.pos] == '_'):
                symbol_str += self.text[self.pos]
                self.pos += 1
            if symbol_str == 'struct':
                return Struct()
            else:
                return Symbol(symbol_str)
        else:
            raise Exception("unsupported token at column %d: %s" %
                            (self.pos + 1, self.get_pos_tip(self.pos)))

    def next_token_expect(self, expect_cls):
        token = self.next_token()
        if not isinstance(token, expect_cls):
            self.backward()
            expect_name = expect_cls.__name__.upper()
            try:
                expect_name = "'%s'" % str(expect_cls())
            except:
                pass
            raise Exception("expect %s at column %d, but get '%s' %s" %
                            (expect_name, self.pos + 1, token,
                             self.get_pos_tip(self.pos)))
        return token

    def get_pos_tip(self, pos):
        while pos < len(self.text) and self.text[pos].isspace():
            pos += 1
        return '\n' + self.text + '\n' + ' ' * pos + '^' + '\n'

    def error_with_last_pos(self, message):
        if self.pos_history:
            column = self.pos_history[-1]
            message = 'error at column %d: ' % (column + 1) + \
                self.get_pos_tip(column) + message
        return Exception(message)

    def __enter__(self):
        self.archived_pos = self.pos
        self.archived_pos_history = self.pos_history[:]

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            self.pos = self.archived_pos
            self.pos_history = self.archived_pos_history


class Dereference(AST):
    def __init__(self, variable, member=None):
        self.variable = variable
        self.member = member

    def __str__(self):
        if self.member:
            return '(%s)->%s' % (self.variable, self.member)
        else:
            return '*%s' % self.variable


class Access(AST):
    def __init__(self, variable, member):
        self.variable = variable
        self.member = member

    def __str__(self):
        return '(%s).%s' % (self.variable, self.member)


class Typecast(AST):
    def __init__(self, variable, new_type, ref_level, keyword, indexes):
        self.variable = variable
        self.new_type = new_type
        self.ref_level = ref_level
        self.keyword = keyword
        self.indexes = indexes

    @property
    def type_str(self):
        return (self.keyword + ' ' if self.keyword else '') + \
                    self.new_type + ' ' + \
                    '*' * self.ref_level + \
                    ''.join(['[%d]' % i for i in self.indexes])

    def __str__(self):
        return '((%s)%s)' % (self.type_str, self.variable)


class Index(AST):
    def __init__(self, variable, index):
        self.variable = variable
        self.index = index

    def __str__(self):
        return '%s[%d]' % (self.variable, self.index)


class Parser(object):
    """
    expr: (LPAREN (STRUCT)? SYMBOL (ASTERISK)* (LSQUARE NUMBER RSQUARE)* RPAREN)? term
    term: (ASTERISK term) | variable (LSQUARE NUMBER RSQUARE | DOT SYMBOL | RARROW SYMBOL)*
    variable: SYMBOL | NUMBER | LPAREN expr RPAREN
    """
    def __init__(self, lexer):
        self.lexer = lexer

    def parse(self):
        expr = self.parse_expr()
        token = self.lexer.next_token()
        if token:
            raise self.lexer.error_with_last_pos(
                "unexpected token '%s' after expression '%s'" % (token, expr))
        return expr

    @log_arg_ret
    def parse_expr(self):
        # try to parse with typecast first
        try:
            with self.lexer:
                keyword = ''
                indexes = []
                ref_level = 0
                self.lexer.next_token_expect(Lparen)
                token = self.lexer.next_token()
                if isinstance(token, Struct):
                    keyword = str(token)
                    symbol = self.lexer.next_token_expect(Symbol)
                elif isinstance(token, Symbol):
                    symbol = token
                else:
                    raise self.lexer.error_with_last_pos(
                        "expect 'struct' or symbol, but get '%s'" % token)

                token = self.lexer.next_token()
                while token:
                    if isinstance(token, Rparen):
                        break
                    elif isinstance(token, Asterisk):
                        ref_level += 1
                    elif isinstance(token, Lsquare):
                        indexes.append(
                            self.lexer.next_token_expect(Number).value)
                        self.lexer.next_token_expect(Rsquare)
                    else:
                        raise self.lexer.error_with_last_pos(
                            "expect '*' or ')', but get '%s'", token)
                    token = self.lexer.next_token()
                if not isinstance(token, Rparen):
                    raise self.lexer.error_with_last_pos(
                        "typecast missing ')'")
                term = self.parse_term()
                return Typecast(term, str(symbol), ref_level, keyword, indexes)
        except:
            pass

        # then try to parse without typecast
        return self.parse_term()

    @log_arg_ret
    def parse_term(self):
        token = self.lexer.next_token()
        if isinstance(token, Asterisk):
            return Dereference(self.parse_term())

        self.lexer.backward()
        term = self.parse_variable()
        token = self.lexer.next_token()
        while token:
            if isinstance(token, Dot):
                term = Access(term, self.lexer.next_token_expect(Symbol))
            elif isinstance(token, Rarrow):
                term = Dereference(term, self.lexer.next_token_expect(Symbol))
            elif isinstance(token, Lsquare):
                number = self.lexer.next_token_expect(Number)
                self.lexer.next_token_expect(Rsquare)
                term = Index(term, number.value)
            else:
                self.lexer.backward()
                break
            token = self.lexer.next_token()

        return term

    @log_arg_ret
    def parse_variable(self):
        token = self.lexer.next_token()
        if isinstance(token, Symbol) or isinstance(token, Number):
            return token
        elif isinstance(token, Lparen):
            expr = self.parse_expr()
            self.lexer.next_token_expect(Rparen)
            return expr
        else:
            raise self.lexer.error_with_last_pos(
                "expect symbol, number or expression, but get '%s'" % token)


def read_timeout(fp, timeout=1, size=1024*1024):
    r, _, _ = select.select([fp], [], [], timeout)
    if fp not in r:
        return b''

    return os.read(fp.fileno(), size)


class GdbShell(object):
    PROMPT = '(gdb)'

    def __init__(self, elf_path):
        self.gdb = subprocess.Popen('gdb ' + elf_path, shell=True,
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, bufsize=1)

        output, err = self._read_output()
        if self.PROMPT not in output or err:
            raise Exception('gdb init failed, path: %s\n'
                            '----- stdout -----\n%s\n'
                            '----- stderr -----\n%s' %
                            (elf_path, output, err))

    def __del__(self):
        gdb = getattr(self, 'gdb', None)
        if not gdb:
            return

        gdb.kill()
        gdb.wait()

    def _read_output(self):
        output = read_timeout(self.gdb.stdout).decode()
        output_all = ''
        while output:
            output_all += output
            if self.PROMPT in output:
                break
            output = read_timeout(self.gdb.stdout).decode()
            # print("output: '%s'" % repr(output))

        err_str = read_timeout(self.gdb.stderr, 0).decode()

        return output_all, err_str

    @log_arg_ret
    def run_cmd(self, cmd):
        self.gdb.stdin.write((cmd + '\n').encode())
        self.gdb.stdin.flush()
        output, err = self._read_output()
        lines = output.splitlines()
        if self.PROMPT not in output or err or len(lines) <= 1:
            raise Exception('run gdb command "%s" failed\n'
                            '----- stdout -----\n%s\n'
                            '----- stderr -----\n%s' %
                            (cmd, output, err))

        return '\n'.join(lines[:-1])


class Dumper(object):
    BLANK = '    '

    def __init__(self, pid, array_max, string_max, hex_string, elf_path=None):
        exe = os.readlink(
                os.path.join('/proc', str(pid), 'exe'))
        self.array_max = array_max
        self.string_max = string_max
        self.hex_string = hex_string
        self.elf_path = exe
        if elf_path:
            self.elf_path = elf_path
        if not os.access(self.elf_path, os.R_OK):
            raise Exception("can't read '%s', "
                            "you can pass '-e ELF_PATH' to solve it" %
                            self.elf_path)

        mem_path = os.path.join('/proc', str(pid), 'mem')
        self.mem_fp = open(mem_path, 'rb')
        maps = open(os.path.join('/proc', str(pid), 'maps')).read()
        self.base_addr = int([line for line in maps.splitlines()
                              if exe in line][0].split('-')[0], base=16)
        self.gdb_shell = GdbShell(self.elf_path)
        self.arch_size = 8
        if platform.architecture()[0] == '32bit':
            self.arch_size = 4

    def simplify_type(self, type_str):
        simplified_str = type_str.strip()
        if simplified_str[0] == '(':
            if simplified_str[-1] != ')':
                raise Exception("parentheses not enclosed: '%s'" % type_str)
            return simplified_str[1:-1]

        return simplified_str

    @log_arg_ret
    def get_member_offset_and_type(self, type_str, member):
        # output = self.gdb_shell.run_cmd('ptype ' + type_str)
        # # example: '   struct _43rdewd ** _4rmem[43][5]'
        # pattern = r'^\s*((\w+\s)?\w+(\s*\*+)?)\s*' + member + r'((\[\d+\])*);'
        # match = re.match(pattern, output)
        # if not match or not match.group(1):
        #     raise Exception("type '%s' has no member '%s', ptype: %s" %
        #                     type_str, member, output)
        # member_type = match.group(1) + (match.group(4) if match.group(4) else '')
        try:
            output = self.gdb_shell.run_cmd('p &((%s *)0)->%s' %
                                            (type_str, member))
            pos1 = output.index('=')
            pos2 = output.index('0x')
            member_type = self.simplify_type(output[pos1 + 1:pos2])
            offset_str = output[pos2:].strip().split()[0]
            member_offset = int(offset_str, 0)
            return member_offset, self.derefrence_type(member_type)[0]
        except Exception as e:
            log_exception()
            raise Exception("failed to get offset(%s, %s): %s" %
                            (type_str, member, str(e)))

    @log_arg_ret
    def get_symbol_address_and_type(self, symbol_str):
        try:
            output = self.gdb_shell.run_cmd('p &%s' % symbol_str)
            pos1 = output.index('=')
            pos2 = output.index('0x')
            symbol_type = self.simplify_type(output[pos1 + 1:pos2])
            symbol_offset = int(output[pos2:].strip().split()[0], 0)
            return symbol_offset + self.base_addr, \
                self.derefrence_type(symbol_type)[0]
        except Exception as e:
            log_exception()
            raise Exception("failed to get address of symbol '%s': %s" %
                            (symbol_str, str(e)))

    @log_arg_ret
    def get_type_size(self, type_str):
        try:
            output = self.gdb_shell.run_cmd('p sizeof(%s)' % type_str)
            pos = output.index('=')
            return int(output[pos + 1:].strip().split()[0], 0)
        except Exception as e:
            log_exception()
            raise Exception("failed to get size of '%s': %s" %
                            (type_str, str(e)))

    @log_arg_ret
    def dereference_addr(self, address):
        self.mem_fp.seek(address)
        try:
            data = self.mem_fp.read(self.arch_size)
        except:
            raise Exception("read at address 0x%x failed" % address)

        return struct.unpack('P', data)[0]

    @log_arg_ret
    def derefrence_type(self, type_str):
        # remove a '(*)' or '[\d+]' or '*'
        if '(*)' in type_str:
            return type_str.replace('(*)', '', 1).strip(), '(*)'

        # example: 'struct _43rdewd ** [43] [5]'
        match = re.match(r'^(\w+\s+)?\w+\s*(\*)*\s*(\[\d+\])?', type_str)
        if match:
            _, group2, group3 = match.groups()
            if group3:
                return type_str.replace(group3, '', 1).strip(), group3
            elif group2:
                return type_str.replace(group2, '', 1).strip(), group2
        raise Exception("type '%s' is neither array nor pointer, "
                        "can't derefrence it", type_str)

    @log_arg_ret
    def get_addr_and_type(self, expr):
        if isinstance(expr, Typecast):
            addr, _ = self.get_addr_and_type(expr.variable)
            return addr, expr.type_str
        elif isinstance(expr, Access):
            addr, type_str = self.get_addr_and_type(expr.variable)
            if '*' in type_str:
                raise Exception("type of '%s' is '%s', '->' "
                                "should be used instead of '.'" %
                                (expr.variable, type_str))
            if '[' in type_str:
                raise Exception("type of '%s' is '%s', not a struct, "
                                "'.' is not allowed" %
                                (expr.variable, type_str))
            offset, type_str = self.get_member_offset_and_type(
                type_str, expr.member)
            return addr + offset, type_str
        elif isinstance(expr, Dereference):
            addr, type_str = self.get_addr_and_type(expr.variable)
            if '*' not in type_str:
                if expr.member:
                    raise Exception("type of '%s' is '%s', '.' "
                                    "should be used instead of '->'" %
                                    (expr.variable, type_str))
                else:
                    raise Exception("type of '%s' is '%s', not a pointer, "
                                    "'*' is not allowed" %
                                    (expr.variable, type_str))
            type_str = self.derefrence_type(type_str)[0]
            addr = self.dereference_addr(addr)
            offset = 0
            if expr.member:
                offset, type_str = self.get_member_offset_and_type(
                    type_str, expr.member)
            return addr + offset, type_str
        elif isinstance(expr, Index):
            addr, type_str = self.get_addr_and_type(expr.variable)
            if '*' not in type_str and '[' not in type_str:
                raise Exception("type of '%s' is '%s', neither pointer "
                                "nor array, index is not allowed" %
                                (expr.variable, type_str))
            type_str, popped = self.derefrence_type(type_str)
            if '*' in popped:
                # index a pointer instead of array
                addr = self.dereference_addr(addr)
            type_size = self.get_type_size(type_str)
            return addr + type_size * expr.index, type_str
        elif isinstance(expr, Symbol):
            return self.get_symbol_address_and_type(expr.value)
        elif isinstance(expr, Number):
            return expr.value, ''

    def dump_type(self, data, type_str, indent):
        dump_txt = ''
        if '*' in type_str:
            # dump pointer
            return '0x%x' % struct.unpack('P', data[:self.arch_size])[0]

        # example: 'struct _43rdewd  [43] [5]'
        match = re.match(r'(^\w+\s+)?\w+\s*\[(\d+)\]', type_str)
        if match:
            # dump array
            type_str = type_str.replace('[%s]' % match.group(2), '')
            type_size = self.get_type_size(type_str)
            count = int(match.group(2))
            omit_count = 0
            if type_size == 1:
                # dump byte array
                omit_tip = ''
                if indent > 0 and count > self.string_max:
                    data = data[:self.string_max]
                    omit_tip = '...'
                if not self.hex_string:
                    try:
                        # dump string
                        str_val = data.decode('ascii')
                        return '"%s"' % (str_val + omit_tip)
                    except:
                        pass
                # dump hex
                return '"<binary>" /* hex: %s */' % \
                       (codecs.encode(data, 'hex').decode() + omit_tip)

            # dump array in each line
            if indent > 0 and count > self.array_max:
                omit_count = count - self.array_max
                count = self.array_max
            dump_txt += '{\n'
            indent += 1
            for i in range(count):
                dump_txt += indent * self.BLANK + self.dump_type(
                    data[type_size * i: type_size * i + type_size],
                    type_str, indent) + ',\n'
            if omit_count:
                dump_txt += indent * self.BLANK + \
                            '// other %s elements are omit\n' %\
                            omit_count
            indent -= 1
            dump_txt += indent * self.BLANK + '}'
            return dump_txt

        try:
            type_desc = self.gdb_shell.run_cmd('ptype %s' % type_str)
            pos = type_desc.index('=')
            type_desc = type_desc[pos + 1:].strip().splitlines()
        except Exception as e:
            log_exception()
            raise Exception("failed to get type of '%s': %s" %
                            (type_str, str(e)))

        # dump basic type
        if len(type_desc) == 1:
            if 'char' in type_desc[0]:
                if 'unsigned' in type_desc[0]:
                    return str(struct.unpack('B', data[0])[0])
                else:
                    return str(struct.unpack('b', data[0])[0])
            elif 'short' in type_desc[0]:
                if 'unsigned' in type_desc[0]:
                    return str(struct.unpack('H', data[0:2])[0])
                else:
                    return str(struct.unpack('h', data[0:2])[0])
            elif 'int' in type_desc[0]:
                if 'unsigned' in type_desc[0]:
                    return str(struct.unpack('I', data[0:4])[0])
                else:
                    return str(struct.unpack('i', data[0:4])[0])
            elif 'long' in type_desc[0]:
                if 'unsigned' in type_desc[0]:
                    return str(struct.unpack('L', data[0:self.arch_size])[0])
                else:
                    return str(struct.unpack('l', data[0:self.arch_size])[0])
            else:
                return "// unsupported type: '%s'" % type_desc[0].strip()

        # dump struct
        dump_txt += '{\n'
        indent += 1
        for line in type_desc:
            if not line or '{' in line or '}' in line:
                continue
            # example: '  struct _43rdewd *foo [43] [5];'
            match = re.match(
                r'^\s*((\w+\s+)?\w+\s+\**)\s*(\w+)((\s*\[\d+\])*);', line)
            if not match or not match.group(3):
                dump_txt += indent * self.BLANK + \
                            "// can't recognize '%s'\n" % line.strip()
            else:
                # member_type = (match.group(1) + match.group(4)).strip()
                member = match.group(3).strip()
                try:
                    offset, member_type = self.get_member_offset_and_type(
                        type_str, member)
                    member_size = self.get_type_size(member_type)
                    dump_txt += indent * self.BLANK + '.' + member + ' = ' + \
                                self.dump_type(
                                    data[offset: offset + member_size],
                                    member_type, indent) + ',\n'
                except:
                    dump_txt += \
                        indent * self.BLANK + \
                        "// parse member '%s' of type '%s' failed\n" % \
                        (member, type_str)
                    log_exception()
        indent -= 1
        dump_txt += indent * self.BLANK + '}'
        return dump_txt

    def dump(self, expr):
        addr, type_str = self.get_addr_and_type(expr)
        type_size = self.get_type_size(type_str)
        self.mem_fp.seek(addr)
        try:
            data = self.mem_fp.read(type_size)
        except:
            raise Exception("read memory 0x%x-0x%x failed" %
                            (addr, addr + type_size))
        return "%s = %s;" % (expr, self.dump_type(data, type_str, indent=0))


if __name__ == '__main__':
    epilog = """examples:
    * type of g_var1 is 'struct foo**', dump the second struct:
        %(prog)s 32311 '*g_var1[2]'
    * type of g_var2 is 'struct foo*', dump the first 5 elements:
        %(prog)s 32311 '(struct foo[5])*g_var2'
    * g_var3 points to a nested struct, dump the member:
        %(prog)s 32311 'g_var3->val.data'
    * there is a 'struct foo' at address 0x556159b32020, dump it:
        %(prog)s 32311 '(struct foo)0x556159b32020'
    """ % {'prog': sys.argv[0]}
    parser = argparse.ArgumentParser(
        description='dump global variables of a living process without interruption it',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog)
    parser.add_argument('pid', type=int, help='target process ID')
    parser.add_argument('expression', type=str, help='a C rvalue expression')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='show debug information')
    parser.add_argument('-x', '--hex-string', action='store_true',
                        help='dump byte array in hex instead of string')
    parser.add_argument('-e', '--elf-path',
                        help='elf path to read symbols, '
                             '`readlink /proc/$pid/exe` by default')
    parser.add_argument('-a', '--array-max', type=int, default=6,
                        help='maximum number of elements to display in an array')
    parser.add_argument('-s', '--string-max', type=int, default=64,
                        help='displayed maximum string length')
    args = parser.parse_args()
    verbose = args.verbose
    try:
        lexer = Lexer(args.expression)
        parser = Parser(lexer)
        expr = parser.parse()
        dumper = Dumper(args.pid, args.array_max, args.string_max,
                        args.hex_string, args.elf_path)
        print(dumper.dump(expr))
    except Exception as e:
        if verbose:
            raise

        print("dump '%s' failed: %s" % (args.expression, str(e)),
              file=sys.stderr)
        exit(-1)
