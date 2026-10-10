# 同梱の声プリセット(crates/engine/assets/voices/<name>.{flac,json})を Gemini TTS で作る。
#
#   .\tools\voice-presets\make_presets.ps1                  # 無いものだけ作る
#   .\tools\voice-presets\make_presets.ps1 -Only genki -Force # 作り直す
#
# API キー: 環境変数 GEMINI_API_KEY → GUI で保存したキー(data/config.json、DPAPI)の順。
# 要 ffmpeg(FLAC へ変換する)。マイク・GPU には触れない。
# プリセットを増やしたら crates/engine/src/presets.rs の PRESETS にも足す。
param([string]$Model = 'gemini-3.8-flash-tts', [string[]]$Only, [switch]$Force)
$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '../..')
$dir = Join-Path $root 'crates/engine/assets/voices'

$key = $env:GEMINI_API_KEY
if (-not $key) {
  Add-Type -AssemblyName System.Security
  $cfg = Get-Content (Join-Path $root 'data/config.json') -Raw | ConvertFrom-Json
  if (-not $cfg.gemini_api_key_protected) { throw 'GEMINI_API_KEY を設定するか、GUI で Gemini の API キーを保存してください' }
  $key = [Text.Encoding]::UTF8.GetString([Security.Cryptography.ProtectedData]::Unprotect(
      [Convert]::FromBase64String($cfg.gemini_api_key_protected), $null, 'CurrentUser'))
}

# voice は Gemini TTS の既定の声。セリフは 10〜13 秒になる長さにする(参照音声の目安)
$presets = @(
  @{ name='tsundere'; voice='Leda';     style='若い女性の声で、ツンデレ気味に、少し照れながら強がって';
     text='べ、別にあなたのために作ったわけじゃないんだからね。たまたま時間が余っただけ。……でも、美味しいって言ってくれたら、ちょっとだけ嬉しいかも。' },
  @{ name='genki';    voice='Zephyr';   style='明るく元気な少女の声で、弾むように楽しそうに';
     text='おはよう！今日はすっごくいい天気だね！ねえねえ、お昼になったら一緒に公園に行こうよ。お弁当もちゃんと二人分作ってきたんだから！' },
  @{ name='oneesan';  voice='Sulafat';  style='落ち着いた大人の女性の声で、優しく包み込むように、ゆったりと';
     text='お疲れさま。今日も一日よく頑張ったわね。温かいお茶を淹れたから、少しここで休んでいきなさい。話なら、いくらでも聞いてあげるから。' },
  @{ name='kuudere';  voice='Achernar'; style='クールで淡々とした若い女性の声で、感情を抑えて静かに';
     text='報告します。目標地点までの距離はおよそ三キロ。天候は安定しています。……別に心配しているわけではありません。ただ、無理はしないでください。' },
  @{ name='shounen';  voice='Puck';     style='元気で生意気な少年の声で、得意げに';
     text='へへっ、見てよこれ！裏山で見つけたんだ、すっげえでっかいカブトムシ！兄ちゃんにも見せてやろうと思って、ずっと走ってきたんだぜ。' },
  @{ name='seinen';   voice='Charon';   style='落ち着いた青年の男性の声で、穏やかに誠実に';
     text='大丈夫、焦らなくていいよ。君のペースで話してくれればいい。僕はここにいるから、ゆっくり考えて、それから一緒に答えを探そう。' },
  @{ name='ojisan';   voice='Algenib';  style='渋く低い中年男性の声で、ゆっくりと重みのある語り口で';
     text='若い頃はな、俺も無茶ばかりしたもんだ。だが、失敗した数だけ見える景色ってのがある。だから怖がるな。一歩踏み出してみろ。' },
  @{ name='narrator'; voice='Kore';     style='ニュートラルで聞き取りやすいナレーターの声で、落ち着いて明瞭に';
     text='この街には、古くから伝わる不思議な言い伝えがあります。満月の夜、時計台の鐘が十三回鳴ったとき、願いがひとつだけ叶うというのです。' }
)

foreach ($p in $presets) {
  if ($Only -and $p.name -notin $Only) { continue }
  $flac = Join-Path $dir "$($p.name).flac"
  if ((Test-Path $flac) -and -not $Force) { "skip (exists): $($p.name)"; continue }
  # 指示はプロンプトの演出メモに書き、TRANSCRIPT だけを読ませる(systemInstruction は TTS モデルで使えず、
  # 「次のセリフを〜読んで」のような地の文は指示ごと読み上げられる)
  $prompt = "# AUDIO PROFILE`nJapanese voice actor.`n`n## DIRECTOR'S NOTES`nStyle: $($p.style)。`nSpeak only the transcript below, nothing else.`n`n## TRANSCRIPT`n$($p.text)"
  $body = @{
    contents = @(@{ role = 'user'; parts = @(@{ text = $prompt }) })
    generationConfig = @{ responseModalities = @('AUDIO'); speechConfig = @{ voiceConfig = @{ prebuiltVoiceConfig = @{ voiceName = $p.voice } } } }
  } | ConvertTo-Json -Depth 10
  $r = Invoke-RestMethod -Method Post -Uri "https://generativelanguage.googleapis.com/v1beta/models/${Model}:generateContent" `
        -Headers @{ 'x-goog-api-key' = $key } -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))
  $inline = ($r.candidates[0].content.parts | Where-Object { $_.inlineData } | Select-Object -First 1).inlineData
  # 返る形式はモデルで違う(WAV ヘッダ付き / 生の PCM16)。ffmpeg に形式を伝えて FLAC にする
  $tmp = Join-Path ([IO.Path]::GetTempPath()) "voice-preset-$($p.name).bin"
  [IO.File]::WriteAllBytes($tmp, [Convert]::FromBase64String($inline.data))
  $inputFmt = if ($inline.mimeType -match 'wav') { @() } else {
    $rate = if ($inline.mimeType -match 'rate=(\d+)') { $Matches[1] } else { '24000' }
    @('-f', 's16le', '-ar', $rate, '-ac', '1')
  }
  & ffmpeg -loglevel error -y @inputFmt -i $tmp -c:a flac -compression_level 12 -sample_fmt s16 $flac
  if ($LASTEXITCODE -ne 0) { throw "ffmpeg failed: $($p.name)" }
  Remove-Item $tmp
  [IO.File]::WriteAllText((Join-Path $dir "$($p.name).json"), "{`n  `"source`": `"${Model}:$($p.voice)`"`n}`n")
  "$($p.name): $($p.voice) ($($inline.mimeType))"
}
