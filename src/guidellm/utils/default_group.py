"""
File uses code adapted from code with the following license:

Copyright (c) 2015-2023, Heungsub Lee
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

  Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

__all__ = ["DefaultGroupHandler"]

import collections.abc as cabc

import click


class DefaultGroupHandler(click.Group):
    """
    Allows the migration to a new sub-command by allowing the group to run
    one of its sub-commands as the no-args default command.
    """

    def __init__(self, *args, **kwargs):
        # To resolve as the default command.
        if not kwargs.get("ignore_unknown_options", True):
            raise ValueError("Default group accepts unknown options")
        self.ignore_unknown_options = True
        self.default_cmd_name = kwargs.pop("default", None)
        self.default_if_no_args = kwargs.pop("default_if_no_args", False)
        super().__init__(*args, **kwargs)

    def parse_args(self, ctx, args):
        if not args and self.default_if_no_args:
            args.insert(0, self.default_cmd_name)
        return super().parse_args(ctx, args)

    def get_command(self, ctx, cmd_name):
        if cmd_name not in self.commands:
            # If it doesn't match an existing command, use the default command name.
            ctx.arg0 = cmd_name
            cmd_name = self.default_cmd_name
        return super().get_command(ctx, cmd_name)

    def resolve_command(self, ctx, args):
        cmd_name, cmd, args = super().resolve_command(ctx, args)
        if hasattr(ctx, "arg0"):
            args.insert(0, ctx.arg0)
            if cmd is not None:
                cmd_name = cmd.name
        return cmd_name, cmd, args

    def format_commands(self, ctx, formatter):
        """
        Used to wrap the default formatter to clarify which command is the default.
        """
        formatter = DefaultCommandFormatter(self, formatter, mark=" (default)")
        return super().format_commands(ctx, formatter)


class DefaultCommandFormatter(click.HelpFormatter):
    """
    Wraps a formatter to edit the line for the default command to mark it
    with the specified mark string.
    """

    def __init__(self, group, formatter, mark="*"):
        self.group = group
        self.formatter = formatter
        self.mark = mark
        super().__init__()

    def __getattr__(self, attr):
        return getattr(self.formatter, attr)

    def write_dl(self, rows: cabc.Iterable[tuple[str, str]], *args, **kwargs):
        rows_: list[tuple[str, str]] = []
        for cmd_name, help_msg in rows:
            if cmd_name == self.group.default_cmd_name:
                rows_.insert(0, (cmd_name + self.mark, help_msg))
            else:
                rows_.append((cmd_name, help_msg))
        return self.formatter.write_dl(rows_, *args, **kwargs)
