; Установщик настольного приложения «Контроль качества DXA» (Inno Setup 6).
;
;   ISCC desktop\installer.iss /DAppVersion=1.0.0
;
; Берёт готовую сборку PyInstaller из dist\DXA-QC и пишет
; dist\DXA-QC-Setup-<версия>.exe. Установка без прав администратора — в
; %LOCALAPPDATA%\Programs\DXA-QC (или для всех пользователей, если выбрать
; это в мастере и подтвердить права).

#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif
#define AppName "Контроль качества DXA"
#define AppExe "DXA-QC.exe"
#define CliExe "dxa-qc-cli.exe"
#define SrcDir "..\dist\DXA-QC"

[Setup]
AppId={{6B8E3F52-1C4A-4E4F-9A57-2D3C8B91DA05}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=Команда проекта DXA QC, ЛЦТ 2026
AppPublisherURL=https://github.com/theorcos239/LCT_2026
AppSupportURL=https://github.com/theorcos239/LCT_2026
DefaultDirName={autopf}\DXA-QC
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir=..\dist
OutputBaseFilename=DXA-QC-Setup-{#AppVersion}
SetupIconFile=assets\icon.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
InfoBeforeFile=installer_info.txt
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
WizardStyle=modern
ChangesEnvironment=yes
CloseApplications=yes
VersionInfoVersion={#AppVersion}
VersionInfoProductName={#AppName}
VersionInfoDescription=Установщик: {#AppName}

[Languages]
Name: "ru"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
ru.BotShortcut=Telegram-бот
en.BotShortcut=Telegram bot
ru.ServeShortcut=Веб-сервер для локальной сети
en.ServeShortcut=Web server for local network
ru.CliShortcut=Командная строка (пакетная обработка)
en.CliShortcut=Command line (batch processing)
ru.AddToPath=Добавить dxa-qc-cli в PATH (пакетная обработка из любой консоли)
en.AddToPath=Add dxa-qc-cli to PATH
ru.NoWebView2=На компьютере нет WebView2 Runtime. Приложение откроет интерфейс в браузере; чтобы работать в отдельном окне, установите WebView2 Runtime с сайта Microsoft.
en.NoWebView2=WebView2 Runtime is not installed. The app will open in a browser instead; install WebView2 Runtime from Microsoft to get a native window.

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "addtopath"; Description: "{cm:AddToPath}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#SrcDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "installer_info.txt"; DestDir: "{app}"; DestName: "ПРОЧТИТЕ.txt"; Flags: ignoreversion
Source: "..\docs\Инструкция_приложение.pdf"; DestDir: "{app}"; DestName: "Инструкция — приложение.pdf"; Flags: ignoreversion skipifsourcedoesntexist
Source: "..\docs\Инструкция_Telegram-бот.pdf"; DestDir: "{app}"; DestName: "Инструкция — Telegram-бот.pdf"; Flags: ignoreversion skipifsourcedoesntexist

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\{cm:BotShortcut}"; Filename: "{app}\{#CliExe}"; Parameters: "bot"; WorkingDir: "{app}"; IconFilename: "{app}\{#AppExe}"
Name: "{group}\{cm:ServeShortcut}"; Filename: "{app}\{#CliExe}"; Parameters: "serve"; WorkingDir: "{app}"; IconFilename: "{app}\{#AppExe}"
Name: "{group}\{cm:CliShortcut}"; Filename: "{cmd}"; Parameters: "/k ""cd /d ""{app}"" && {#CliExe} --help"""; WorkingDir: "{app}"; IconFilename: "{app}\{#AppExe}"
Name: "{group}\ПРОЧТИТЕ"; Filename: "{app}\ПРОЧТИТЕ.txt"
Name: "{group}\Инструкция — приложение"; Filename: "{app}\Инструкция — приложение.pdf"; Check: ManualInstalled('Инструкция — приложение.pdf')
Name: "{group}\Инструкция — Telegram-бот"; Filename: "{app}\Инструкция — Telegram-бот.pdf"; Check: ManualInstalled('Инструкция — Telegram-бот.pdf')
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; Check: NeedsAddPath(ExpandConstant('{app}')); Tasks: addtopath

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; кэш окна, журналы, токен бота — всё, что приложение создало само
Type: filesandordirs; Name: "{localappdata}\DXA-QC"
Type: filesandordirs; Name: "{userappdata}\DXA-QC"

[Code]
{ Ярлык инструкции создаётся, только если её PDF попал в установщик }
function ManualInstalled(Name: string): Boolean;
begin
  Result := FileExists(ExpandConstant('{app}\') + Name);
end;

function NeedsAddPath(Dir: string): Boolean;
var
  Paths: string;
begin
  if not RegQueryStringValue(HKCU, 'Environment', 'Path', Paths) then
  begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Uppercase(Dir) + ';', ';' + Uppercase(Paths) + ';') = 0;
end;

procedure RemoveFromPath(Dir: string);
var
  Paths: string;
  P: Integer;
begin
  if not RegQueryStringValue(HKCU, 'Environment', 'Path', Paths) then
    exit;
  P := Pos(';' + Uppercase(Dir), Uppercase(Paths));
  if P > 0 then
  begin
    Delete(Paths, P, Length(Dir) + 1);
    RegWriteExpandStringValue(HKCU, 'Environment', 'Path', Paths);
  end;
end;

function HasWebView2: Boolean;
var
  V: string;
begin
  Result :=
    RegQueryStringValue(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', V) or
    RegQueryStringValue(HKLM, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', V) or
    RegQueryStringValue(HKCU, 'Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', V);
  Result := Result and (V <> '') and (V <> '0.0.0.0');
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and (not HasWebView2) and (not WizardSilent) then
    MsgBox(CustomMessage('NoWebView2'), mbInformation, MB_OK);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    RemoveFromPath(ExpandConstant('{app}'));
end;
