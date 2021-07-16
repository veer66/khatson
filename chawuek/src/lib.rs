use anyhow::Result;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use tch::jit::IValue;
use tch::CModule;
use tch::Tensor;
use thiserror::Error;

#[allow(dead_code)]
pub fn cargo_dir() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

#[derive(Error, Debug)]
pub enum ChawuekError {
    #[error("Cannot open the character map file")]
    CannotOpenCharMapFile,
    #[error("Cannot parse the character map")]
    CannotParseCharMapFile,
    #[error("Cannot find special symbol `{0}` in the character map")]
    CannotFindSpecialSymbolInCharMap(String),
    #[error("Cannot get character from string `{0}`")]
    CannotCharFromString(String),
    #[error("The module returned an invalid value.")]
    ModuleReturnedAnInvalidValue,
}
#[derive(Debug)]
struct CharToXi {
    punc_i: i64,
    unk_i: i64,
    pad_i: i64,
    char_ix_map: HashMap<char, i64>,
}

impl CharToXi {
    fn get_special_symbol(imm: &HashMap<String, i64>, sym: &str) -> Result<i64> {
        Ok(*imm
            .get(sym)
            .ok_or_else(|| ChawuekError::CannotFindSpecialSymbolInCharMap(sym.to_owned()))?)
    }

    pub fn load_char_map() -> Result<CharToXi> {
        let char_map_path = Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/data/attacut-c/characters.json"
        ));
        let f = File::open(char_map_path).map_err(|_| ChawuekError::CannotOpenCharMapFile)?;
        let reader = BufReader::new(f);
        let imm: HashMap<String, i64> =
            serde_json::from_reader(reader).map_err(|_| ChawuekError::CannotParseCharMapFile)?;
        let punc_i = Self::get_special_symbol(&imm, "<PUNC>")?;
        let pad_i = Self::get_special_symbol(&imm, "<PAD>")?;
        let unk_i = Self::get_special_symbol(&imm, "<UNK>")?;
        let mut char_ix_map: HashMap<char, i64> = HashMap::new();
        for (k, v) in imm {
            let k: Vec<char> = k.chars().collect();
            if k.len() == 1 {
                char_ix_map.insert(k[0], v);
            }
        }
        Ok(CharToXi {
            punc_i,
            pad_i,
            unk_i,
            char_ix_map,
        })
    }

    pub fn to_xi(&self, ch: &char) -> i64 {
        if ch.is_ascii_punctuation() {
            self.punc_i
        } else {
            *self.char_ix_map.get(ch).unwrap_or(&self.unk_i)
        }
    }
}

pub struct Chawuek {
    model: CModule,
    char_to_xi: CharToXi,
}

impl Chawuek {
    pub fn new() -> Result<Chawuek> {
        let model_path = Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/data/attacut-c/model.pt"
        ));
        let model = CModule::load(model_path)?;
        let char_to_xi = CharToXi::load_char_map()?;
        Ok(Chawuek { model, char_to_xi })
    }

    pub fn tokenize(&self, txt: &str) -> Result<Vec<String>> {
        let chars: Vec<char> = txt.chars().collect();
        let ch_ix: Vec<i64> = chars.iter().map(|ch| self.char_to_xi.to_xi(&ch)).collect();
        let features = Tensor::of_slice(&ch_ix).view([1, -1]);
        let seq_lengths = [ch_ix.len() as i64];
        let seq_lengths = Tensor::of_slice(&seq_lengths);
        let ivalue = IValue::Tuple(vec![IValue::Tensor(features), IValue::Tensor(seq_lengths)]);
        let out = self.model.forward_is(&[ivalue]).unwrap();
        let pred_threshold = 0.5;
        if let IValue::Tensor(out) = out {
            let probs = Vec::<f32>::from(out.sigmoid());
            let mut buf = String::new();
            buf.push(chars[0]);
            let mut toks = vec![];
            for (p, ch) in probs.into_iter().zip(chars.iter()).skip(1) {
                if p > pred_threshold {
                    toks.push(buf);
                    buf = String::new();
                }
                buf.push(*ch);
            }
            toks.push(buf);
            Ok(toks)
        } else {
            Err(anyhow::Error::new(
                ChawuekError::ModuleReturnedAnInvalidValue,
            ))
        }
    }
}
