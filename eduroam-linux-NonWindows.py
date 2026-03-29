#!/usr/bin/env python3
"""
 * **************************************************************************
 * Contributions to this work were made on behalf of the GÉANT project,
 * a project that has received funding from the European Union’s Framework
 * Programme 7 under Grant Agreements No. 238875 (GN3)
 * and No. 605243 (GN3plus), Horizon 2020 research and innovation programme
 * under Grant Agreements No. 691567 (GN4-1) and No. 731122 (GN4-2).
 * On behalf of the aforementioned projects, GEANT Association is
 * the sole owner of the copyright in all material which was developed
 * by a member of the GÉANT project.
 * GÉANT Vereniging (Association) is registered with the Chamber of
 * Commerce in Amsterdam with registration number 40535155 and operates
 * in the UK as a branch of GÉANT Vereniging.
 *
 * Registered office: Hoekenrode 3, 1102BR Amsterdam, The Netherlands.
 * UK branch address: City House, 126-130 Hills Road, Cambridge CB2 1PQ, UK
 *
 * License: see the web/copyright.inc.php file in the file structure or
 *          <base_url>/copyright.php after deploying the software

Authors:
    Tomasz Wolniewicz <twoln@umk.pl>
    Michał Gasewicz <genn@umk.pl> (Network Manager support)

Contributors:
    Steffen Klemer https://github.com/sklemer1
    ikreb7 https://github.com/ikreb7
    Dimitri Papadopoulos Orfanos https://github.com/DimitriPapadopoulos
    sdasda7777 https://github.com/sdasda7777
    Matt Jolly http://gitlab.com/Matt.Jolly
Many thanks for multiple code fixes, feature ideas, styling remarks
much of the code provided by them in the form of pull requests
has been incorporated into the final form of this script.

This script is the main body of the CAT Linux installer.
In the generation process configuration settings are added
as well as messages which are getting translated into the language
selected by the user.

The script runs under python3.

"""
import argparse
import base64
import getpass
import os
import platform
import re
import subprocess
import sys
import uuid
from shutil import copyfile
from typing import List, Optional, Type, Union

NM_AVAILABLE = True
NEW_CRYPTO_AVAILABLE = True
OPENSSL_CRYPTO_AVAILABLE = False
DEBUG_ON = False

parser = argparse.ArgumentParser(description='eduroam linux installer.')
parser.add_argument('--debug', '-d', action='store_true', dest='debug',
                    default=False, help='set debug flag')
parser.add_argument('--username', '-u', action='store', dest='username',
                    help='set username')
parser.add_argument('--password', '-p', action='store', dest='password',
                    help='set text_mode flag')
parser.add_argument('--silent', '-s', action='store_true', dest='silent',
                    help='set silent flag')
parser.add_argument('--pfxfile', action='store', dest='pfx_file',
                    help='set path to user certificate file')
parser.add_argument("--wpa_conf", action='store_true', dest='wpa_conf',
                    help='generate wpa_supplicant config file without configuring the system')
parser.add_argument("--iwd_conf", action='store_true', dest='iwd_conf',
                    help='generate iwd config file without configuring the system')
parser.add_argument("--gui", action='store', dest='gui',
                    help='one of: tty, tkinter, zenity, kdialog, yad - use this GUI system if present, falling back to standard choice if not')
ARGS = parser.parse_args()
if ARGS.debug:
    DEBUG_ON = True
    print("Running debug mode")


def debug(msg) -> None:
    """Print debugging messages to stdout"""
    if not DEBUG_ON:
        return
    print("DEBUG:" + str(msg))


def byte_to_string(barray: List) -> str:
    """conversion utility"""
    return "".join([chr(x) for x in barray])

def join_with_separator(list: List, separator: str) -> str:
    """
    Join a list of strings with a separator; if only one element
    return that element unchanged.
    """
    if len(list) > 1:
        return separator.join(list)
    if list:
        return list[0]
    return ""

debug(sys.version_info.major)

try:
    import dbus
except ImportError:
    print("WARNING: Cannot import the dbus module for "+sys.executable+" - please install dbus-python!")
    debug("Cannot import the dbus module for "+sys.executable)
    NM_AVAILABLE = False


try:
    from cryptography.hazmat.primitives.serialization import pkcs12
    from cryptography.hazmat.backends import default_backend
    from cryptography.x509.oid import NameOID
except ImportError:
    NEW_CRYPTO_AVAILABLE = False
    try:
        from OpenSSL import crypto
        crypto.load_pkcs12  # missing in newer versions
        OPENSSL_CRYPTO_AVAILABLE = True
    except (ImportError, AttributeError):  # AttributeError sometimes thrown by old/broken OpenSSL versions
        OPENSSL_CRYPTO_AVAILABLE = False


def detect_desktop_environment() -> str:
    """
    Detect what desktop type is used. This method is prepared for
    possible future use with password encryption on supported distros

    the function below was partially copied from
    https://ubuntuforums.org/showthread.php?t=1139057
    """
    desktop_environment = 'generic'
    if os.environ.get('KDE_FULL_SESSION') == 'true':
        desktop_environment = 'kde'
    elif os.environ.get('GNOME_DESKTOP_SESSION_ID'):
        desktop_environment = 'gnome'
    else:
        try:
            shell_command = subprocess.Popen(['xprop', '-root',
                                              '_DT_SAVE_MODE'],
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE)
            out, _ = shell_command.communicate()
            info = out.decode('utf-8').strip()
        except (OSError, RuntimeError):
            pass
        else:
            if ' = "xfce4"' in info:
                desktop_environment = 'xfce'
    return desktop_environment


def get_system() -> List:
    """
    Detect Linux platform. Not used at this stage.
    It is meant to enable password encryption in distros
    that can handle this well.
    """
    system = platform.system_alias(
        platform.system(),
        platform.release(),
        platform.version()
    )
    return [system, detect_desktop_environment()]


def get_config_path() -> str:
    """
    Return XDG_CONFIG_HOME path if exists otherwise $HOME/.config
    """

    xdg_config_home_path = os.environ.get('XDG_CONFIG_HOME')
    if not xdg_config_home_path:
        home_path = os.environ.get('HOME')
        return f'{home_path}/.config'
    return xdg_config_home_path


def run_installer() -> None:
    """
    This is the main installer part. It tests for NM availability
    gets user credentials and starts a proper installer.
    """
    global ARGS
    global NM_AVAILABLE
    username = ''
    password = ''
    silent = False
    pfx_file = ''
    gui = ''
    wpa_conf = False
    iwd_conf = False

    if ARGS.username:
        username = ARGS.username
    if ARGS.password:
        password = ARGS.password
    if ARGS.silent:
        silent = ARGS.silent
    if ARGS.pfx_file:
        pfx_file = ARGS.pfx_file
    if ARGS.wpa_conf:
        wpa_conf = ARGS.wpa_conf
    if ARGS.iwd_conf:
        iwd_conf = ARGS.iwd_conf
    if ARGS.gui:
        gui = ARGS.gui
    debug(get_system())
    debug("Calling InstallerData")
    installer_data = InstallerData(silent=silent, username=username,
                                   password=password, pfx_file=pfx_file, gui=gui)

    if wpa_conf:
        NM_AVAILABLE = False
    if iwd_conf:
        NM_AVAILABLE = False

    # test dbus connection
    if NM_AVAILABLE:
        config_tool = CatNMConfigTool()
        if config_tool.connect_to_nm() is None:
            NM_AVAILABLE = False
    if not NM_AVAILABLE and not wpa_conf and not iwd_conf:
        # no dbus so ask if the user will want wpa_supplicant config
        if installer_data.ask(Messages.save_wpa_conf, Messages.cont, 1):
            sys.exit(1)
    installer_data.get_user_cred()
    installer_data.save_ca()
    if NM_AVAILABLE:
        config_tool.add_connections(installer_data)
    elif iwd_conf:
        iwd_config = IwdConfiguration()
        for ssid in Config.ssids:
            iwd_config.generate_iwd_config(ssid, installer_data)
    else:
        wpa_config = WpaConf()
        wpa_config.create_wpa_conf(Config.ssids, installer_data)
    installer_data.show_info(Messages.installation_finished)


class Messages:
    """
    These are initial definitions of messages, but they will be
    overridden with translated strings.
    """
    quit = "Really quit?"
    credentials_prompt = "Please, enter your credentials:"
    username_prompt = "enter your userid"
    enter_password = "enter password"
    enter_import_password = "enter your import password"
    incorrect_password = "incorrect password"
    repeat_password = "repeat your password"
    passwords_differ = "passwords do not match"
    empty_field = "one of the fields was empty"
    installation_finished = "Installation successful"
    cat_dir_exists = "Directory {} exists; some of its files may be " \
                     "overwritten."
    cont = "Continue?"
    nm_not_supported = "This NetworkManager version is not supported"
    cert_error = "Certificate file not found, looks like a CAT error"
    unknown_version = "Unknown version"
    dbus_error = "DBus connection problem, a sudo might help"
    ok = "OK"
    yes = "Y"
    nay = "N"
    p12_filter = "personal certificate file (p12 or pfx)"
    all_filter = "All files"
    p12_title = "personal certificate file (p12 or pfx)"
    save_wpa_conf = "NetworkManager configuration failed. " \
                    "Ensure you have the dbus-python package for your distro installed on your system. " \
                    "We may generate a wpa_supplicant configuration file " \
                    "if you wish. Be warned that your connection password will be saved " \
                    "in this file as clear text."
    save_wpa_confirm = "Write the file"
    wrongUsernameFormat = "Error: Your username must be of the form " \
                          "'xxx@institutionID' e.g. 'john@example.net'!"
    wrong_realm = "Error: your username must be in the form of 'xxx@{}'. " \
                  "Please enter the username in the correct format."
    wrong_realm_suffix = "Error: your username must be in the form of " \
                         "'xxx@institutionID' and end with '{}'. Please enter the username " \
                         "in the correct format."
    user_cert_missing = "personal certificate file not found"
    # "File %s exists; it will be overwritten."
    # "Output written to %s"


class Config:
    """
    This is used to prepare settings during installer generation.
    """
    instname = ""
    profilename = ""
    url = ""
    email = ""
    title = "eduroam CAT"
    servers = []
    ssids = []
    del_ssids = []
    eap_outer = ''
    eap_inner = ''
    use_other_tls_id = False
    server_match = ''
    anonymous_identity = ''
    CA = ""
    init_info = ""
    init_confirmation = ""
    tou = ""
    sb_user_file = ""
    verify_user_realm_input = False
    user_realm = ""
    hint_user_input = False


class InstallerData:
    """
    General user interaction handling, supports zenity, KDialog, yad and
    standard command-line interface
    """

    def __init__(self, silent: bool = False, username: str = '',
                 password: str = '', pfx_file: str = '', gui: str = '') -> None:
        self.graphics = ''
        self.username = username
        self.password = password
        self.silent = silent
        self.pfx_file = pfx_file
        if gui in ('tty', 'tkinter', 'yad', 'zenity', 'kdialog'):
            self.gui = gui
        else:
            self.gui = ''
        debug("starting constructor")
        if silent:
            self.graphics = 'tty'
        else:
            self.__get_graphics_support()
        self.show_info(Config.init_info.format(Config.instname,
                                               Config.email, Config.url))
        if self.ask(Config.init_confirmation.format(Config.instname,
                                                    Config.profilename),
                    Messages.cont, 1):
            sys.exit(1)
        if Config.tou != '':
            if self.ask(Config.tou, Messages.cont, 1):
                sys.exit(1)
        if os.path.exists(get_config_path() + '/cat_installer'):
            if self.ask(Messages.cat_dir_exists.format(
                    get_config_path() + '/cat_installer'),
                    Messages.cont, 1):
                sys.exit(1)
        else:
            os.mkdir(get_config_path() + '/cat_installer', 0o700)

    @staticmethod
    def save_ca() -> None:
        """
        Save CA certificate to cat_installer directory
        (create directory if needed)
        """
        certfile = get_config_path() + '/cat_installer/ca.pem'
        debug("saving cert")
        with open(certfile, 'w') as cert:
            cert.write(Config.CA + "\n")

    def ask(self, question: str, prompt: str = '', default: Optional[bool] = None) -> int:
        """
        Prompt user for a Y/N reply, possibly supplying a default answer
        """
        if self.silent:
            return 0
        if self.graphics == 'tty':
            yes = Messages.yes[:1].upper()
            nay = Messages.nay[:1].upper()
            print("\n-------\n" + question + "\n")
            while True:
                tmp = prompt + " (" + Messages.yes + "/" + Messages.nay + ") "
                if default == 1:
                    tmp += "[" + yes + "]"
                elif default == 0:
                    tmp += "[" + nay + "]"
                inp = input(tmp)
                if inp == '':
                    if default == 1:
                        return 0
                    if default == 0:
                        return 1
                i = inp[:1].upper()
                if i == yes:
                    return 0
                if i == nay:
                    return 1
        elif self.graphics == 'tkinter':
            from tkinter import messagebox
            return 0 if messagebox.askyesno(Config.title, question + "\n\n" + prompt) else 1
        else:
            command = []
            if self.graphics == "zenity":
                command = ['zenity', '--title=' + Config.title, '--width=500',
                           '--question', '--text=' + question + "\n\n" + prompt]
            elif self.graphics == 'kdialog':
                command = ['kdialog', '--yesno', question + "\n\n" + prompt,
                           '--title=' + Config.title]
            elif self.graphics == 'yad':
                command = ['yad', '--image="dialog-question"',
                           '--button=gtk-yes:0',
                           '--button=gtk-no:1',
                           '--width=500',
                           '--wrap',
                           '--text=' + question + "\n\n" + prompt,
                           '--title=' + Config.title]
            return subprocess.call(command, stderr=subprocess.DEVNULL)

    def show_info(self, data: str) -> None:
        """
        Show a piece of information
        """
        if self.silent:
            return
        if self.graphics == 'tty':
            print(data)
        elif self.graphics == 'tkinter':
            from tkinter import messagebox
            messagebox.showinfo(Config.title, data)
        else:
            if self.graphics == "zenity":
                command = ['zenity', '--info', '--width=500', '--text=' + data]
            elif self.graphics == "kdialog":
                command = ['kdialog', '--msgbox', data, '--title=' + Config.title]
            elif self.graphics == "yad":
                command = ['yad', '--button=OK', '--width=500', '--text=' + data]
            else:
                sys.exit(1)
            subprocess.call(command, stderr=subprocess.DEVNULL)

    def confirm_exit(self) -> None:
        """
        Confirm exit from installer
        """
        ret = self.ask(Messages.quit)
        if ret == 0:
            sys.exit(1)

    def alert(self, text: str) -> None:
        """Generate alert message"""
        if self.silent:
            return
        if self.graphics == 'tty':
            print(text)
        elif self.graphics == 'tkinter':
            from tkinter import messagebox
            messagebox.showwarning(Config.title, text)
        else:
            if self.graphics == 'zenity':
                command = ['zenity', '--warning', '--text=' + text]
            elif self.graphics == "kdialog":
                command = ['kdialog', '--sorry', text, '--title=' + Config.title]
            elif self.graphics == "yad":
                command = ['yad', '--text=' + text]
            else:
                sys.exit(1)
            subprocess.call(command, stderr=subprocess.DEVNULL)

    def prompt_nonempty_string(self, show: int, prompt: str, val: str = '') -> str:
        """
        Prompt user for input
        """
        if self.graphics == 'tty':
            if show == 0:
                while True:
                    inp = str(getpass.getpass(prompt + ": "))
                    output = inp.strip()
                    if output != '':
                        return output
            while True:
                inp = input(prompt + ": ")
                output = inp.strip()
                if output != '':
                    return output
        elif self.graphics == 'tkinter':
            from tkinter import simpledialog
            while True:
                output = simpledialog.askstring(Config.title, prompt,
                                                initialvalue=val,
                                                show="*" if show == 0 else "")
                if output:
                    return output

        else:
            command = []
            if self.graphics == 'zenity':
                if val == '':
                    default_val = ''
                else:
                    default_val = '--entry-text=' + val
                if show == 0:
                    hide_text = '--hide-text'
                else:
                    hide_text = ''
                command = ['zenity', '--entry', hide_text, default_val,
                           '--width=500', '--text=' + prompt]
            elif self.graphics == 'kdialog':
                if show == 0:
                    hide_text = '--password'
                else:
                    hide_text = '--inputbox'
                command = ['kdialog', hide_text, prompt, '--title=' + Config.title]
            elif self.graphics == 'yad':
                if show == 0:
                    hide_text = ':H'
                else:
                    hide_text = ''
                command = ['yad', '--form', '--field=' + hide_text,
                           '--text=' + prompt, val]

            output = ''
            while not output:
                shell_command = subprocess.Popen(command, stdout=subprocess.PIPE,
                                                 stderr=subprocess.PIPE)
                out, _ = shell_command.communicate()
                output = out.decode('utf-8')
                if self.graphics == 'yad':
                    output = output[:-2]
                output = output.strip()
                if shell_command.returncode == 1:
                    self.confirm_exit()
            return output

    def __get_username_password_atomic(self) -> None:
        """
        use single form to get username, password and password confirmation
        """
        output_fields_separator = "\n\n\n\n\n"
        while True:
            password = "a"
            password1 = "b"
            if self.graphics == 'tkinter':
                import tkinter as tk

                root = tk.Tk()
                root.title(Config.title)

                desc_label = tk.Label(root, text=Messages.credentials_prompt)
                desc_label.grid(row=0, column=0, columnspan=2, sticky=tk.W)

                username_label = tk.Label(root, text=Messages.username_prompt)
                username_label.grid(row=1, column=0, sticky=tk.W)

                username_entry = tk.Entry(root, textvariable=tk.StringVar(root, value=self.username))
                username_entry.grid(row=1, column=1)

                password_label = tk.Label(root, text=Messages.enter_password)
                password_label.grid(row=2, column=0, sticky=tk.W)

                password_entry = tk.Entry(root, show="*")
                password_entry.grid(row=2, column=1)

                password1_label = tk.Label(root, text=Messages.repeat_password)
                password1_label.grid(row=3, column=0, sticky=tk.W)

                password1_entry = tk.Entry(root, show="*")
                password1_entry.grid(row=3, column=1)

                def submit(installer):
                    def inner():
                        nonlocal password, password1
                        (installer.username, password, password1) = (username_entry.get(), password_entry.get(), password1_entry.get())
                        root.destroy()
                    return inner

                login_button = tk.Button(root, text=Messages.ok, command=submit(self))
                login_button.grid(row=4, column=0, columnspan=2)

                root.mainloop()
            else:
                if self.graphics == 'zenity':
                    command = ['zenity', '--forms', '--width=500',
                               f"--title={Config.title}",
                               f"--text={Messages.credentials_prompt}",
                               f"--add-entry={Messages.username_prompt}",
                               f"--add-password={Messages.enter_password}",
                               f"--add-password={Messages.repeat_password}",
                               "--separator", output_fields_separator]
                elif self.graphics == 'yad':
                    command = ['yad', '--form',
                               f"--title={Config.title}",
                               f"--text={Messages.credentials_prompt}",
                               f"--field={Messages.username_prompt}", self.username,
                               f"--field={Messages.enter_password}:H",
                               f"--field={Messages.repeat_password}:H",
                               "--separator", output_fields_separator]

                output = ''
                while not output:
                    shell_command = subprocess.Popen(command, stdout=subprocess.PIPE,
                                                     stderr=subprocess.PIPE)
                    out, _ = shell_command.communicate()
                    output = out.decode('utf-8')
                    if self.graphics == 'yad':
                        output = output[:-(len(output_fields_separator)+1)]
                    output = output.strip()
                    if shell_command.returncode == 1:
                        self.confirm_exit()

                if self.graphics in ('zenity', 'yad'):
                    fields = output.split(output_fields_separator)
                    if len(fields) != 3:
                        self.alert(Messages.empty_field)
                        continue
                    self.username, password, password1 = fields

            if not self.__validate_user_name():
                continue
            if password != password1:
                self.alert(Messages.passwords_differ)
                continue
            self.password = password
            break

    def get_user_cred(self) -> None:
        """
        Get user credentials both username/password and personal certificate
        based
        """
        if Config.eap_outer in ('PEAP', 'TTLS'):
            self.__get_username_password()
        elif Config.eap_outer == 'TLS':
            self.__get_p12_cred()

    def __get_username_password(self) -> None:
        """
        read user password and set the password property
        do nothing if silent mode is set
        """
        if self.silent:
            return
        if self.graphics in ('tkinter', 'zenity', 'yad'):
            self.__get_username_password_atomic()
        else:
            password = "a"
            password1 = "b"
            if self.username:
                user_prompt = self.username
            elif Config.hint_user_input:
                user_prompt = '@' + Config.user_realm
            else:
                user_prompt = ''
            while True:
                self.username = self.prompt_nonempty_string(
                    1, Messages.username_prompt, user_prompt)
                if self.__validate_user_name():
                    break
            while password != password1:
                password = self.prompt_nonempty_string(
                    0, Messages.enter_password)
                password1 = self.prompt_nonempty_string(
                    0, Messages.repeat_password)
                if password != password1:
                    self.alert(Messages.passwords_differ)
            self.password = password

    def __check_graphics(self, command) -> bool:
        shell_command = subprocess.Popen(['which', command],
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)
        shell_command.wait()
        if shell_command.returncode == 0:
            self.graphics = command
            debug("Using "+command)
            return True
        return False

    def __get_graphics_support(self) -> None:
        self.graphics = 'tty'
        if self.gui == 'tty':
            return
        if os.environ.get('DISPLAY') is None:
            return
        if self.gui != 'tkinter':
            if self.__check_graphics(self.gui):
                return
            try:
                import tkinter  # noqa: F401
            except Exception:
                pass
            else:
                self.graphics = 'tkinter'
                return
            for cmd in ('yad', 'zenity', 'kdialog'):
                if self.__check_graphics(cmd):
                    return

    def __process_p12(self) -> bool:
        debug('process_p12')
        pfx_file = get_config_path() + '/cat_installer/user.p12'
        if NEW_CRYPTO_AVAILABLE:
            debug("using new crypto")
            try:
                p12 = pkcs12.load_key_and_certificates(
                                        open(pfx_file,'rb').read(),
                                        self.password, backend=default_backend())
            except Exception as error:
                debug(f"Incorrect password ({error}).")
                return False
            else:
                if Config.use_other_tls_id:
                    return True
                try:
                    self.username = p12[1].subject.\
                        get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
                except crypto.Error:
                    self.username = p12[1].subject.\
                        get_attributes_for_oid(NameOID.EMAIL_ADDRESS)[0].value
                return True
        if OPENSSL_CRYPTO_AVAILABLE:
            debug("using openssl crypto")
            try:
                p12 = crypto.load_pkcs12(open(pfx_file, 'rb').read(),
                                         self.password)
            except crypto.Error as error:
                debug(f"Incorrect password ({error}).")
                return False
            else:
                if Config.use_other_tls_id:
                    return True
                try:
                    self.username = p12.get_certificate(). \
                        get_subject().commonName
                except crypto.Error:
                    self.username = p12.get_certificate().\
                        get_subject().emailAddress
                return True
        debug("using openssl")
        command = ['openssl', 'pkcs12', '-in', pfx_file, '-passin',
                   'pass:' + self.password, '-nokeys', '-clcerts']
        shell_command = subprocess.Popen(command, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)
        out, _ = shell_command.communicate()
        if shell_command.returncode != 0:
            debug("first password run failed")
            command1 = ['openssl', 'pkcs12', '-legacy', '-in', pfx_file, '-passin',
                        'pass:' + self.password, '-nokeys', '-clcerts']
            shell_command1 = subprocess.Popen(command1, stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE)
            out, err = shell_command1.communicate()
            if shell_command1.returncode != 0:
                return False
        if Config.use_other_tls_id:
            return True
        out_str = out.decode('utf-8').strip()
        # split only on commas that are not inside double quotes
        subject = re.split(r'\s*[/,]\s*(?=([^"]*"[^"]*")*[^"]*$)',
                           re.findall(r'subject=/?(.*)$',
                                      out_str, re.MULTILINE)[0])
        cert_prop = {}
        for field in subject:
            if field:
                cert_field = re.split(r'\s*=\s*', field)
                cert_prop[cert_field[0].lower()] = cert_field[1]
        if cert_prop['cn'] and re.search(r'@', cert_prop['cn']):
            debug('Using cn: ' + cert_prop['cn'])
            self.username = cert_prop['cn']
        elif cert_prop['emailaddress'] and \
                re.search(r'@', cert_prop['emailaddress']):
            debug('Using email: ' + cert_prop['emailaddress'])
            self.username = cert_prop['emailaddress']
        else:
            self.username = ''
            self.alert("Unable to extract username "
                       "from the certificate")
        return True

    def __select_p12_file(self) -> str:
        """
        prompt user for the PFX file selection
        this method is not being called in the silent mode
        therefore there is no code for this case
        """
        if self.graphics == 'tty':
            my_dir = os.listdir(".")
            p_count = 0
            pfx_file = ''
            for my_file in my_dir:
                if my_file.endswith(('.p12', '*.pfx', '.P12', '*.PFX')):
                    p_count += 1
                    pfx_file = my_file
            prompt = "personal certificate file (p12 or pfx)"
            default = ''
            if p_count == 1:
                default = '[' + pfx_file + ']'

            while True:
                inp = input(prompt + default + ": ")
                output = inp.strip()

                if default != '' and output == '':
                    return pfx_file
                default = ''
                if os.path.isfile(output):
                    return output
                print("file not found")

        cert = ""
        if self.graphics == 'zenity':
            command = ['zenity', '--file-selection',
                       '--file-filter=' + Messages.p12_filter +
                       ' | *.p12 *.P12 *.pfx *.PFX', '--file-filter=' +
                       Messages.all_filter + ' | *',
                       '--title=' + Messages.p12_title]
            shell_command = subprocess.Popen(command, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE)
            cert, _ = shell_command.communicate()
        if self.graphics == 'kdialog':
            command = ['kdialog', '--getopenfilename', '.',
                       '--title=' + Messages.p12_title]
            shell_command = subprocess.Popen(command, stdout=subprocess.PIPE,
                                             stderr=subprocess.DEVNULL)
            cert, _ = shell_command.communicate()
        if self.graphics == 'yad':
            command = ['yad', '--file',
                       '--file-filter=*.p12 *.P12 *.pfx *.PFX',
                       '-file-filter=*', '--title=' + Messages.p12_title]
            shell_command = subprocess.Popen(command, stdout=subprocess.PIPE,
                                             stderr=subprocess.DEVNULL)
            cert, _ = shell_command.communicate()
        if self.graphics == 'tkinter':
            from tkinter import filedialog as fd
            return fd.askopenfilename(title=Messages.p12_title,
                                      filetypes=(("Certificate file",
                                                  ("*.p12", "*.P12", "*.pfx",
                                                   "*.PFX")),))
        return cert.decode('utf-8').strip()

    @staticmethod
    def __save_sb_pfx() -> None:
        """write the user PFX file"""
        cert_file = get_config_path() + '/cat_installer/user.p12'
        with open(cert_file, 'wb') as cert:
            cert.write(base64.b64decode(Config.sb_user_file))

    def __get_p12_cred(self):
        """get the password for the PFX file"""
        if Config.eap_inner == 'SILVERBULLET':
            self.__save_sb_pfx()
        else:
            if not self.silent:
                self.pfx_file = self.__select_p12_file()
            try:
                copyfile(self.pfx_file, get_config_path() +
                         '/cat_installer/user.p12')
            except (OSError, RuntimeError):
                print(Messages.user_cert_missing)
                sys.exit(1)
        if self.silent:
            username = self.username
            if not self.__process_p12():
                sys.exit(1)
            if username:
                self.username = username
        else:
            while not self.password:
                self.password = self.prompt_nonempty_string(
                    0, Messages.enter_import_password).encode('utf-8')
                if not self.__process_p12():
                    self.alert(Messages.incorrect_password)
                    self.password = ''
            if not self.username:
                self.username = self.prompt_nonempty_string(
                    1, Messages.username_prompt)

    def __validate_user_name(self) -> bool:
        # locate the @ character in username
        pos = self.username.find('@')
        debug("@ position: " + str(pos))
        # trailing @
        if pos == len(self.username) - 1:
            debug("username ending with @")
            self.alert(Messages.wrongUsernameFormat)
            return False
        # no @ at all
        if pos == -1:
            if Config.verify_user_realm_input:
                debug("missing realm")
                self.alert(Messages.wrongUsernameFormat)
                return False
            debug("No realm, but possibly correct")
            return True
        # @ at the beginning
        if pos == 0:
            debug("missing user part")
            self.alert(Messages.wrongUsernameFormat)
            return False
        pos += 1
        if Config.verify_user_realm_input:
            if Config.hint_user_input:
                if self.username.endswith('@' + Config.user_realm, pos - 1):
                    debug("realm equal to the expected value")
                    return True
                debug("incorrect realm; expected:" + Config.user_realm)
                self.alert(Messages.wrong_realm.format(Config.user_realm))
                return False
            if self.username.endswith(Config.user_realm, pos):
                debug("realm ends with expected suffix")
                return True
            debug("realm suffix error; expected: " + Config.user_realm)
            self.alert(Messages.wrong_realm_suffix.format(
                Config.user_realm))
            return False
        pos1 = self.username.find('@', pos)
        if pos1 > -1:
            debug("second @ character found")
            self.alert(Messages.wrongUsernameFormat)
            return False
        pos1 = self.username.find('.', pos)
        if pos1 == pos:
            debug("a dot immediately after the @ character")
            self.alert(Messages.wrongUsernameFormat)
            return False
        debug("all passed")
        return True


class WpaConf:
    """
    Prepare and save wpa_supplicant config file
    """

    @staticmethod
    def __prepare_network_block(ssid: str, user_data: Type[InstallerData]) -> str:
        interface = """network={
        ssid=\"""" + ssid + """\"
        key_mgmt=WPA-EAP
        pairwise=CCMP
        group=CCMP TKIP
        eap=""" + Config.eap_outer + """
        ca_cert=\"""" + get_config_path() + """/cat_installer/ca.pem\"""""""
        identity=\"""" + user_data.username + """\"""""""
        altsubject_match=\"""" + ";".join(Config.servers) + """\"
        """

        if Config.eap_outer in ('PEAP', 'TTLS'):
            interface += f"phase2=\"auth={Config.eap_inner}\"\n" \
                         f"\tpassword=\"{user_data.password}\"\n"
            if Config.anonymous_identity != '':
                interface += f"\tanonymous_identity=\"{Config.anonymous_identity}\"\n"

        elif Config.eap_outer == 'TLS':
            interface += "\tprivate_key_passwd=\"{}\"\n" \
                         "\tprivate_key=\"{}/.cat_installer/user.p12" \
                         .format(user_data.password, os.environ.get('HOME'))
        interface += "\n}"
        return interface

    def create_wpa_conf(self, ssids, user_data: Type[InstallerData]) -> None:
        """Create and save the wpa_supplicant config file"""
        wpa_conf = get_config_path() + '/cat_installer/cat_installer.conf'
        with open(wpa_conf, 'w') as conf:
            for ssid in ssids:
                net = self.__prepare_network_block(ssid, user_data)
                conf.write(net)

class IwdConfiguration:
    """ support the iNet wireless daemon by Intel """
    def __init__(self):
        self.config = ""

    def write_config(self, ssid: str) -> None:
        """
        Write out an iWD config for a given SSID;
        if permissions are insufficient (e.g. wayland),
        use pkexec to elevate a shell and echo the config into the file.
        """

        file_path = f'{ssid}.8021x'
        try:
            with open(file_path, 'w') as config_file:
                debug("writing: "+file_path)
                config_file.write(self.config)
        except PermissionError:
            command = f'echo "{self.config}" > {file_path}'
            subprocess.run(['pkexec', 'bash', '-c', command], check=True)

    def set_domain_mask() -> str:
        """
        Set the domain mask for the IWD config.
        This is a list of DNS servers that the client will accept
        """
        # The domain mask is a list of DNS servers that the client will accept
        # It is a comma-separated list of DNS servers (without the DNS: prefix)
        domainMask = []

        # We need to strip the DNS: prefix
        for server in Config.servers:
            if server.startswith('DNS:'):
                domainMask.append(server[4:])
            else:
                domainMask.append(server)

        return join_with_separator(domainMask, ':')

    def generate_iwd_config(self, ssid: str, user_data: Type[InstallerData]) -> None:
        """Generate an appropriate IWD 8021x config for a given EAP method"""
        #TODO: It would probably be best to generate these configs from scratch but the logic is a little harder
        #      This would add flexibility when dealing with inner and outer config types.
        if Config.eap_outer == 'PWD':
            self._create_eap_pwd_config(ssid, user_data)
        elif Config.eap_outer == 'PEAP':
            self._create_eap_peap_config(ssid, user_data)
        elif Config.eap_outer == 'TTLS':
            self._create_ttls_pap_config(ssid, user_data)
        else:
            raise ValueError('Invalid connection type')
        self.write_config(ssid)

    def _create_eap_pwd_config(self, ssid: str, user_data: Type[InstallerData]) -> None:
        """ create EAP-PWD configuration """
        self.config = f"""
        [Security]
        EAP-Method=PWD
        EAP-Identity={user_data.username}
        EAP-Password={user_data.password}

        [Settings]
        AutoConnect=True
        """

    def _create_eap_peap_config(self, ssid: str, user_data: Type[InstallerData]) -> None:
        """ create EAP-PEAP configuration """
        if Config.anonymous_identity != '':
            outer_identity = Config.anonymous_identity
        else:
            outer_identity = user_data.username
        self.config = f"""
[Security]
EAP-Method=PEAP
EAP-Identity={outer_identity}
EAP-PEAP-CACert=embed:eduroam_ca_cert
EAP-PEAP-ServerDomainMask={IwdConfiguration.set_domain_mask()}
EAP-PEAP-Phase2-Method=MSCHAPV2
EAP-PEAP-Phase2-Identity={user_data.username}
EAP-PEAP-Phase2-Password={user_data.password}

[Settings]
AutoConnect=true

[@pem@eduroam_ca_cert]
{Config.CA}
"""

    def _create_ttls_pap_config(self, ssid: str, user_data: Type[InstallerData]) -> None:
        """ create TTLS-PAP configuration"""
        if Config.anonymous_identity != '':
            outer_identity = Config.anonymous_identity
        else:
            outer_identity = user_data.username
        self.config = f"""
[Security]
EAP-Method=TTLS
EAP-Identity={outer_identity}
EAP-TTLS-CACert=embed:eduroam_ca_cert
EAP-TTLS-ServerDomainMask={IwdConfiguration.set_domain_mask()}
EAP-TTLS-Phase2-Method=Tunneled-PAP
EAP-TTLS-Phase2-Identity={user_data.username}
EAP-TTLS-Phase2-Password={user_data.password}

[Settings]
AutoConnect=true

[@pem@eduroam_ca_cert]
{Config.CA}
"""

class CatNMConfigTool:
    """
    Prepare and save NetworkManager configuration
    """

    def __init__(self):
        self.cacert_file = None
        self.settings_service_name = None
        self.connection_interface_name = None
        self.system_service_name = "org.freedesktop.NetworkManager"
        self.nm_version = None
        self.pfx_file = None
        self.settings = None
        self.user_data = None
        self.bus = None

    def connect_to_nm(self) -> Union[bool, None]:
        """
        connect to DBus
        """
        try:
            self.bus = dbus.SystemBus()
        except AttributeError:
            # since dbus existed but is empty we have an empty package
            # this gets shipped by pyqt5
            print("DBus not properly installed")
            return None
        except dbus.exceptions.DBusException:
            print("Can't connect to DBus")
            return None
        # check NM version
        self.__check_nm_version()
        debug("NM version: " + self.nm_version)
        if self.nm_version in ("0.9", "1.0"):
            self.settings_service_name = self.system_service_name
            self.connection_interface_name = \
                "org.freedesktop.NetworkManager.Settings.Connection"
            # settings proxy
            sysproxy = self.bus.get_object(
                self.settings_service_name,
                "/org/freedesktop/NetworkManager/Settings")
            # settings interface
            self.settings = dbus.Interface(sysproxy, "org.freedesktop."
                                                     "NetworkManager.Settings")
        elif self.nm_version == "0.8":
            self.settings_service_name = "org.freedesktop.NetworkManager"
            self.connection_interface_name = "org.freedesktop.NetworkMana" \
                                             "gerSettings.Connection"
            # settings proxy
            sysproxy = self.bus.get_object(
                self.settings_service_name,
                "/org/freedesktop/NetworkManagerSettings")
            # settings interface
            self.settings = dbus.Interface(
                sysproxy, "org.freedesktop.NetworkManagerSettings")
        else:
            print(Messages.nm_not_supported)
            return None
        debug("NM connection worked")
        return True

    def __check_opts(self) -> None:
        """
        set certificate files paths and test for existence of the CA cert
        """
        self.cacert_file = get_config_path() + '/cat_installer/ca.pem'
        self.pfx_file = get_config_path() + '/cat_installer/user.p12'
        if not os.path.isfile(self.cacert_file):
            print(Messages.cert_error)
            sys.exit(2)

    def __check_nm_version(self) -> None:
        """
        Get the NetworkManager version
        """
        try:
            proxy = self.bus.get_object(
                self.system_service_name, "/org/freedesktop/NetworkManager")
            props = dbus.Interface(proxy, "org.freedesktop.DBus.Properties")
            version = props.Get("org.freedesktop.NetworkManager", "Version")
        except dbus.exceptions.DBusException:
            version = ""
        if re.match(r'^1\.', version):
            self.nm_version = "1.0"
            return
        if re.match(r'^0\.9', version):
            self.nm_version = "0.9"
            return
        if re.match(r'^0\.8', version):
            self.nm_version = "0.8"
            return
        self.nm_version = Messages.unknown_version

    def __delete_existing_connection(self, ssid: str) -> None:
        """
        checks and deletes earlier connection
        """
        try:
            conns = self.settings.ListConnections()
        except dbus.exceptions.DBusException:
            print(Messages.dbus_error)
            sys.exit(3)
        for each in conns:
            con_proxy = self.bus.get_object(self.system_service_name, each)
            connection = dbus.Interface(
                con_proxy,
                "org.freedesktop.NetworkManager.Settings.Connection")
            try:
                connection_settings = connection.GetSettings()
                if connection_settings['connection']['type'] == '802-11-' \
                                                                'wireless':
                    conn_ssid = byte_to_string(
                        connection_settings['802-11-wireless']['ssid'])
                    if conn_ssid == ssid:
                        debug("deleting connection: " + conn_ssid)
                        connection.Delete()
            except dbus.exceptions.DBusException:
                pass

    def __add_connection(self, ssid: str) -> None:
        debug("Adding connection: " + ssid)
        server_alt_subject_name_list = dbus.Array(Config.servers)
        server_name = Config.server_match
        if self.nm_version in ("0.9", "1.0"):
            match_key = 'altsubject-matches'
            match_value = server_alt_subject_name_list
        else:
            match_key = 'subject-match'
            match_value = server_name
        s_8021x_data = {
            'eap': [Config.eap_outer.lower()],
            'identity': self.user_data.username,
            'ca-cert': dbus.ByteArray(
                f"file://{self.cacert_file}\0".encode()),
            match_key: match_value}
        if Config.eap_outer in ('PEAP', 'TTLS'):
            s_8021x_data['password'] = self.user_data.password
            s_8021x_data['phase2-auth'] = Config.eap_inner.lower()
            if Config.anonymous_identity != '':
                s_8021x_data['anonymous-identity'] = Config.anonymous_identity
            s_8021x_data['password-flags'] = 1
        elif Config.eap_outer == 'TLS':
            s_8021x_data['client-cert'] = dbus.ByteArray(
                f"file://{self.pfx_file}\0".encode())
            s_8021x_data['private-key'] = dbus.ByteArray(
                f"file://{self.pfx_file}\0".encode())
            s_8021x_data['private-key-password'] = self.user_data.password
            s_8021x_data['private-key-password-flags'] = 1
        s_con = dbus.Dictionary({
            'type': '802-11-wireless',
            'uuid': str(uuid.uuid4()),
            'permissions': ['user:' + os.environ.get('USER')],
            'id': ssid
        })
        s_wifi = dbus.Dictionary({
            'ssid': dbus.ByteArray(ssid.encode('utf8')),
            'security': '802-11-wireless-security'
        })
        s_wsec = dbus.Dictionary({
            'key-mgmt': 'wpa-eap',
            'proto': ['rsn'],
            'pairwise': ['ccmp'],
            'group': ['ccmp', 'tkip']
        })
        s_8021x = dbus.Dictionary(s_8021x_data)
        s_ip4 = dbus.Dictionary({'method': 'auto'})
        s_ip6 = dbus.Dictionary({'method': 'auto'})
        con = dbus.Dictionary({
            'connection': s_con,
            '802-11-wireless': s_wifi,
            '802-11-wireless-security': s_wsec,
            '802-1x': s_8021x,
            'ipv4': s_ip4,
            'ipv6': s_ip6
        })
        self.settings.AddConnection(con)

    def add_connections(self, user_data: Type[InstallerData]):
        """Delete and then add connections to the system"""
        self.__check_opts()
        self.user_data = user_data
        for ssid in Config.ssids:
            self.__delete_existing_connection(ssid)
            self.__add_connection(ssid)
        for ssid in Config.del_ssids:
            self.__delete_existing_connection(ssid)


Messages.quit = "Really quit?"
Messages.username_prompt = "enter your userid"
Messages.enter_password = "enter password"
Messages.enter_import_password = "enter your import password"
Messages.credentials_prompt = "Please enter your credentials:"
Messages.incorrect_password = "incorrect password"
Messages.repeat_password = "repeat your password"
Messages.passwords_differ = "passwords do not match"
Messages.empty_field = "one of the fields was empty"
Messages.installation_finished = "Installation successful"
Messages.cat_dir_exisits = "Directory {} exists; some of its files may " \
    "be overwritten."
Messages.cont = "Continue?"
Messages.nm_not_supported = "This NetworkManager version is not " \
    "supported"
Messages.cert_error = "Certificate file not found, looks like a CAT " \
    "error"
Messages.unknown_version = "Unknown version"
Messages.dbus_error = "DBus connection problem, a sudo might help"
Messages.yes = "Y"
Messages.no = "N"
Messages.ok = "OK"
Messages.p12_filter = "personal certificate file (p12 or pfx)"
Messages.all_filter = "All files"
Messages.p12_title = "personal certificate file (p12 or pfx)"
Messages.save_wpa_conf = "DBus module not found - please install " \
    "dbus-python! NetworkManager configuration failed, but we can generate " \
    "a wpa_supplicant configuration file if you wish. Be warned that your " \
    "connection password will be saved in this file as clear text."
Messages.save_wpa_confirm = "Write the file"
Messages.wrongUsernameFormat = "Error: Your username must be of the " \
    "form 'xxx@institutionID' e.g. 'john@example.net'!"
Messages.wrong_realm = "Error: your username must be in the form of " \
    "'xxx@{}'. Please enter the username in the correct format."
Messages.wrong_realm_suffix = "Error: your username must be in the " \
    "form of 'xxx@institutionID' and end with '{}'. Please enter the " \
    "username in the correct format."
Messages.user_cert_missing = "personal certificate file not found"
Messages.cat_dir_exists = "Directory {} exists; some of its files may " \
    "be overwritten"
Config.instname = "DTU - Technical University of Denmark"
Config.profilename = "DTU Non Windows (rev 2022-10-04)"
Config.url = "your local eduroam® support page"
Config.email = "itservice@dtu.dk"
Config.title = "eduroam CAT"
Config.server_match = "win.dtu.dk"
Config.eap_outer = "PEAP"
Config.eap_inner = "MSCHAPV2"
Config.init_info = "This installer has been prepared for {0}\n\nMore " \
    "information and comments:\n\nEMAIL: {1}\nWWW: {2}\n\nInstaller created " \
    "with software from the GEANT project."
Config.init_confirmation = "This installer will only work properly if " \
    "you are a member of {0} and the user group: {1}."
Config.user_realm = "dtu.dk"
Config.ssids = ['eduroam']
Config.del_ssids = []
Config.servers = ['DNS:ait-pisepsn03.win.dtu.dk', 'DNS:ait-pisepsn04.win.dtu.dk']
Config.use_other_tls_id = False
Config.anonymous_identity = "anonymous@dtu.dk"
Config.hint_user_input = True
Config.verify_user_realm_input = True
Config.tou = ""
Config.CA = """-----BEGIN CERTIFICATE-----
MIIFszCCA5ugAwIBAgIQGPyTPfToyJJPRg/BlCoZMjANBgkqhkiG9w0BAQsFADBO
MQswCQYDVQQGEwJESzEmMCQGA1UEChMdRGFubWFya3MgVGVrbmlza2UgVW5pdmVy
c2l0ZXQxFzAVBgNVBAMTDkRUVSBST09UIENBIDAxMB4XDTE1MTIwMjExMDQ0OFoX
DTQwMTIwMjExMTQ0OFowTjELMAkGA1UEBhMCREsxJjAkBgNVBAoTHURhbm1hcmtz
IFRla25pc2tlIFVuaXZlcnNpdGV0MRcwFQYDVQQDEw5EVFUgUk9PVCBDQSAwMTCC
AiIwDQYJKoZIhvcNAQEBBQADggIPADCCAgoCggIBANDKEUG7jDTxJW9L2FeE4nV7
5ejarzkrkRz8wpmK+jpA/IqowG1yi/TDk77yastCBLnt0J7GhbUestDx27QcgpmS
kVNM3F6JAxCFSmswmtOTHRwC1Vp9q7RunVi3fH5NZB8n5d/KnRmS6qpq9xoNxR+1
B+J1dd5EopqfYynwCimyYdXVmuJSeRC/mLm7N5/8PPiggrQSboHc1hA1S63s4oow
ME7mFogdS5tn4k+TBgT75q48zGEzJ2p8HeoMH5h/t+t7UJWuM7valf0dyjKvtIYu
Fe9hdPIX6mtFmhBJBZR3a82UG8Vc1WOKBVZtvAA+YjXgSFyPOIfTtTLSideS0dbV
Hs5fSIIkLsQo+qZ+BOBIwgVj8KH4Tzds/c7YKLeXLQAzZn3hJ8ShZb77ZTn5YAMw
sUmJPRUWyMJZxuBhLNty4GfX58D628ELgZdk8gCxr8okt0G8gMMWiFGNbXPM+p1e
z2qla8toHNvz4FKjbV1Wo303Qk0VPxT5iIF7l4voAIFwRmdlrYy1aU9auvE+E3km
84kzkY68V8Rxt/Ig+1dUmngSFyS81VWndpbPzKZtwMHlaFrtxPVlAQiI7y4vvUtU
GYUdscHe736/itpipfyOk8Y+bvtdKei2AFynUu7nfe1ylz21jZ4LFZ4ICxXldCHJ
eW4EuAll5JBRdOJ09G0vAgMBAAGjgYwwgYkwCwYDVR0PBAQDAgGGMA8GA1UdEwEB
/wQFMAMBAf8wHQYDVR0OBBYEFEGHGrJtr9H49tSOY1yMgdi7Pk7FMBIGCSsGAQQB
gjcVAQQFAgMBAAEwEQYDVR0gBAowCDAGBgRVHSAAMCMGCSsGAQQBgjcVAgQWBBTB
yFfc2YDjEQRdRBoMdYAnYLCu1DANBgkqhkiG9w0BAQsFAAOCAgEAb1L5CcG3w+rd
WHsjxtu19tsLJjwjhfYezADbw8HXKnOcaP9fLrPDRP3YHIJK/LSOYHn2z2Ltb6wl
rDB+0l1WhTyUIVluNXKmbeeQ7KhmvAhZXnCbZ2ibodaRndSHRc62c4jIoUtyHgzb
2PT4nGXZ3UAfSJUhpIDXf9d/B8HVD4PBbqCHeB+16Dd4DusxC+n8jW9yCLWqFfrp
C/7D3nueSOzBAqc0hx5f9zWffI99AN9hNSUn9u9TsFOhyvbYtVAelO+cQeN5uXjc
vMJu44j6tbaJJCmZir9cvst//fRQKe/FW7E4xpQ/PYI4/OhY++xY3VtDFKk8YWwj
kPVX14ZThvFgLTUIpfc6XCDGJD1QYbdXoONKsdzVngv5KSPXxDGR7E85q/HKExvm
8bJteqBMEr9brBjVpT4SJruSoEwT1DU7mITJU0s84SMPgGX0W0Z04EgavJgfwb8V
3dVcdOSfRGI9O3P9u7TFHDLXFBkOcbJEq9+q7fZ2NWQ+ahV/0Vp0XOLxsueRk1J8
WaW2hKLjhHMEAvIq4fvID+CwOaKwa91Q9e4QjffG385IAA2St2iYV1qfC7Gw9rnG
G0If+819pZ0HnHtKUlzOAz4Yh7gbPIQFNObDKQTT6rZrL2fJLoZ5kk0/g4BKfGFd
3mihppQSBG7qQF84ErbTO80Pn2Il7L8=
-----END CERTIFICATE-----
-----BEGIN CERTIFICATE-----
MIIGdzCCBF+gAwIBAgIKYRCS3AABAAAACTANBgkqhkiG9w0BAQsFADBOMQswCQYD
VQQGEwJESzEmMCQGA1UEChMdRGFubWFya3MgVGVrbmlza2UgVW5pdmVyc2l0ZXQx
FzAVBgNVBAMTDkRUVSBST09UIENBIDAxMB4XDTE1MTIwMjExMTkxMVoXDTI3MTIw
MjExMjkxMVowcDESMBAGCgmSJomT8ixkARkWAmRrMRMwEQYKCZImiZPyLGQBGRYD
ZHR1MRMwEQYKCZImiZPyLGQBGRYDd2luMTAwLgYDVQQDEydBZmRlbGluZ2VuIGZv
ciBJVCBTZXJ2aWNlIElzc3VpbmcgQ0EgMDIwggEiMA0GCSqGSIb3DQEBAQUAA4IB
DwAwggEKAoIBAQCjfR055ycQHow0JsvgrywYMFnrf0ETzBQ3qhyW4R87m/KOQgBv
Mn/q3lFGMpFabSxv2auTBe4ZKwOyVbIW1dLNtwBDUZ0Ix1LUUdOlwi83YqmGBObe
rT7hUmNFvaykDjnizszjLpHIxydsdK368u4oclCTPS2Lb5eMMhanwRNVpDtyeoPB
TA3hw/yq9yaDqv49D7diqCPxAC6rwTkjTirs4On8y6WSqiRSDP656XMo6NhTk8f1
dy+8zCvHih7tgzvrAReReR3bbPVx8v3ZIRcRSoKXLXP3wU3bPjHBuOJgSZoI7U+b
tFq9XIwxWG77PDe7OyGx11297d995CL8CrU7AgMBAAGjggIzMIICLzASBgkrBgEE
AYI3FQEEBQIDAQABMCMGCSsGAQQBgjcVAgQWBBTMZ8ENgxEXj672axmHA73ZdGc6
vTAdBgNVHQ4EFgQUBhPbV1NxrI24r7VdZ487d3Ld3D8wgdoGA1UdIASB0jCBzzCB
xAYLKwYBBAHYXIN9AwEwgbQwgYYGCCsGAQUFBwICMHoeeABEAGEAbgBtAGEAcgBr
AHMAIABUAGUAawBuAGkAcwBrAGUAIABVAG4AaQB2AGUAcgBzAGkAdABlAHQAIABD
AGUAcgB0AGkAZgBpAGMAYQB0AGUAIABQAHIAYQBjAHQAaQBjAGUAIABTAHQAYQB0
AGUAbQBlAG4AdDApBggrBgEFBQcCARYdaHR0cDovL3BraS53aW4uZHR1LmRrL3Bv
bGljeS8wBgYEVR0gADAZBgkrBgEEAYI3FAIEDB4KAFMAdQBiAEMAQTALBgNVHQ8E
BAMCAYYwDwYDVR0TAQH/BAUwAwEB/zAfBgNVHSMEGDAWgBRBhxqyba/R+PbUjmNc
jIHYuz5OxTBCBgNVHR8EOzA5MDegNaAzhjFodHRwOi8vcGtpLndpbi5kdHUuZGsv
RFRVJTIwUk9PVCUyMENBJTIwMDEoMSkuY3JsMFoGCCsGAQUFBwEBBE4wTDBKBggr
BgEFBQcwAoY+aHR0cDovL3BraS53aW4uZHR1LmRrL1dJTi1ST09UQ0EwMV9EVFUl
MjBST09UJTIwQ0ElMjAwMSgxKS5jcnQwDQYJKoZIhvcNAQELBQADggIBAK62o90Z
QCDB4hsFRi9IoyrgL8fTJS3PTTXSsdnyRoXAQJzzAsWvvg4iTIMjJmpnYffB07Ax
mAmfJ7mueWVqZ7S0TwZjqgIZJmzzYV44eLn6CUq5Ua5UwaLCv+gsVnz/lR43BWCT
/heKHq6W64ST2whi4f/uhlaQj5zgsMXPtBgLDRsEvXUlrVHilaU7/4PtheeRGdbY
hAXnN6qCJlOeZIrgVtvBqG8hoe4f5pqXsJ4hPRKYxBcA1RI1tb6Z20L3f5+ppqNM
MbOqBTbtRL1IZl0ktLouiOo9/s9rTnDxaFotWp370mGbTqaOuNIxHfhuJC/koaTf
Z3MyMBduQKRh8UzTrM+vkkYww8kG2+ZvAvUl3v6Co27kl37MGleJtxjNsejLx9A5
XKSU29pMG/dHtPWRjlBOZXKuGzcs6TzY1i/HPxmGXn2xmXe4Zxt3akJTZJStZ5xu
4afLprlCYR9Wc7w5FUG6WkrvWBZD9r6UYuQQSknK5KqdL2rymI/4Dp0IYE1ykZXX
P6DFULwVXIypQVwRY2L+JxBJ8EeUEc8LciJjhKFHf2zYwh2B27zDTIcEMXZPvZ42
JaWb94x0JkaiKwPGwTO/Qf//yLhpkhTTat1HmfpsQsd8GQosAdG7DmGT2b84Ps5T
mj11TwBgoKu/qe7tW3wijRQABbjO7EUCtRYq
-----END CERTIFICATE-----
-----BEGIN CERTIFICATE-----
MIIGkTCCBHmgAwIBAgITEwAAAAtIGAItF3lvPQABAAAACzANBgkqhkiG9w0BAQsF
ADBOMQswCQYDVQQGEwJESzEmMCQGA1UEChMdRGFubWFya3MgVGVrbmlza2UgVW5p
dmVyc2l0ZXQxFzAVBgNVBAMTDkRUVSBST09UIENBIDAxMB4XDTIyMTAwMzEyMDEx
M1oXDTM0MTAwMzEyMTExM1owcDESMBAGCgmSJomT8ixkARkWAmRrMRMwEQYKCZIm
iZPyLGQBGRYDZHR1MRMwEQYKCZImiZPyLGQBGRYDd2luMTAwLgYDVQQDEydBZmRl
bGluZ2VuIGZvciBJVCBTZXJ2aWNlIElzc3VpbmcgQ0EgMDMwggIiMA0GCSqGSIb3
DQEBAQUAA4ICDwAwggIKAoICAQCnr3H0nthghzIccgBx0tyPbPk6HM+plbCfeWpV
ATTBALAtP02j9KYYujm3HLV5Bmo+flWqBZRx237SKoTQEEHFE/bbkNuX1Np/U/HP
TyNeY3Hz6v25FrdgcGrrlmaZWA7b3UByV2Iyhe/vSFnGuBOBuUOXlohINnfORCtp
kK3IUfYgefMwNNL0/j8wepYSP/FEb81RBD4Rbas8mVbNczBhvrqxFeifYivTXOg8
PeqL6BbhjNLvNza9EfSunFZdeLzhuIX7KaRgUp06ltBI1pKybJSbuA7cmMos/T/D
Dxk6AX4MrWCF8mbxqjcEU5bAcopAoSjt1zFtC//W8j3QU134ehOszJkog9cXjLl5
Y7hhzmH20mwo3ZVqdPfE2hUrXJPIzzDocsfJzimMxo3D/YqOKbMw9hy1k2a1Q63G
lvRiYte/1at0YlmDbqtpxjH2eZiWzkzIOXvFGlk3AvkwWmMU3IHNgMEwUnPqjczi
LYwfahq75vFOvg5wJkDfChn4wws34BnRpcZQfeP5c3zlKwXALricnf4NDXBXstn/
/sSKXsWcE8O1aFCjBHhEklZfmgP87hQA+owLixWZsXYGV2AbOYut7wMp2slZhpB9
jRCpJ3ux90doJhiFj5XlYmg0cKdUK6VfhhuUDow1I3303eqSui+3q6qo4PbVkOOO
tmOJuQIDAQABo4IBRDCCAUAwEAYJKwYBBAGCNxUBBAMCAQAwHQYDVR0OBBYEFH8x
Erzof/fcUmXmp9PGeUji0vwRMBEGA1UdIAQKMAgwBgYEVR0gADAZBgkrBgEEAYI3
FAIEDB4KAFMAdQBiAEMAQTALBgNVHQ8EBAMCAYYwEgYDVR0TAQH/BAgwBgEB/wIB
ADAfBgNVHSMEGDAWgBRBhxqyba/R+PbUjmNcjIHYuz5OxTBCBgNVHR8EOzA5MDeg
NaAzhjFodHRwOi8vcGtpLndpbi5kdHUuZGsvRFRVJTIwUk9PVCUyMENBJTIwMDEo
MSkuY3JsMFkGCCsGAQUFBwEBBE0wSzBJBggrBgEFBQcwAoY9aHR0cDovL3BraS53
aW4uZHR1LmRrL2FpdC1wcm9vdGNhX0RUVSUyMFJPT1QlMjBDQSUyMDAxKDEpLmNy
dDANBgkqhkiG9w0BAQsFAAOCAgEAVUkOay5rKJcBZCcw3OjnZOT0AhlR8FOTDyzB
CEpmTNGw+6o03jxzRDw2htx6CUKg0rcqu42ajWfMpznD+45BkTBUfBcwdVGvQ0A5
fagKpdZJqjX8h0AubIiVQT+WEVIXLXWYqzLjHKZAOPjh3/c1wXnpfcupMiqUfHyW
PuyQuWk3e2ffD8fqQkXmm5kGhxnYRVwdjBRI1OgwHu+g9y+aMPxDjy6UV9dszbzG
rzp0WUfYP5Po5Q20WisuSP2cslCLWEA1puJ9eoQbolX0lU4akir2+BeeFOymJ4Zr
0sJDV07NiJFQek0KvYQTJ2AoHomSxuMK4JnovfUy5CkSv/c79TT4YCM+j/XMvktT
7JNMhw7RI/+pNLksDDp4G2y4sUR8F6rP9taHfNbrf9hCei7e6+ZV9iP2esWGRy0m
9soeZ0I6PdnFowhlIPI9IiL5oJn9MSS/IS8kJtQi+GEJAZi1skMQwe1JPKdwBlMX
6+N4zcymRlFSxzP9Ff9zsc4eOyw6VrKuVol+5+YzOFC1mjpTrKmNsnQoyLPbaDM+
iLI/+waFbFu1yqTnfOuue/P8+TEfujz/4bwZq3s25mLQH/puEI7ueb1XTxVcJzj6
GF8PvBE+A6iD8oAg+h+3AqsWqGp+3Lr1kGK/5JKw2CXV3SwA3v827uOQ731lwbTK
wQA2RQg=
-----END CERTIFICATE-----
"""


if __name__ == '__main__':
    run_installer()
