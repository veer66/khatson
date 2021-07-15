use serde_json;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use anyhow::Result;
use thiserror::Error;
use tch::Tensor;
use tch::CModule;
use tch::jit::IValue;
#[derive(Error, Debug)]
pub enum KhatsonError {
    #[error("Cannot open the character map file")]
    CannotOpenCharMapFile,
    #[error("Cannot parse the character map")]
    CannotParseCharMapFile,
    #[error("Cannot find special symbol `{0}` in the character map")]
    CannotFindSpecialSymbolInCharMap(String),
    #[error("Cannot get character from string `{0}`")]
    CannotCharFromString(String),


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
        Ok(*imm.get(sym).ok_or_else(|| KhatsonError::CannotFindSpecialSymbolInCharMap(sym.to_owned()))?)
    }
    
    pub fn load_char_map() -> Result<CharToXi> {
        let f = File::open("data/attacut-c/characters.json").map_err(|_| KhatsonError::CannotOpenCharMapFile)?;
        let reader = BufReader::new(f);
        let imm: HashMap<String, i64> = serde_json::from_reader(reader).map_err(|_| KhatsonError::CannotParseCharMapFile)?;
        let punc_i = Self::get_special_symbol(&imm, "<PUNC>")?;
        let pad_i = Self::get_special_symbol(&imm, "<PAD>")?;
        let unk_i = Self::get_special_symbol(&imm, "<UNK>")?;
        let mut char_ix_map: HashMap<char, i64> = HashMap::new();
        for (k,v) in imm {
            let k: Vec<char> = k.chars().collect();
            if k.len() == 1 {
                char_ix_map.insert(k[0], v);
            }
        }
        Ok(CharToXi { punc_i, pad_i, unk_i, char_ix_map })
    }

    pub fn to_xi(&self, ch: &char) -> i64 {
        if ch.is_ascii_punctuation() {
             self.punc_i
        } else {
            *self.char_ix_map.get(ch).unwrap_or(&self.unk_i)
        }
    }
}

fn main() {
    let char_to_xi = CharToXi::load_char_map().unwrap();
    let txt = "กินข้าวม้า";
    let chars: Vec<char> = txt.chars().collect();
    let ch_ix: Vec<i64> = chars.iter().map(|ch| char_to_xi.to_xi(&ch)).collect();
    let features = Tensor::of_slice(&ch_ix).view([1, -1]);
    let seq_lengths = [ch_ix.len() as i64];
    let seq_lengths = Tensor::of_slice(&seq_lengths);
    let model = CModule::load("data/attacut-c/model.pt").unwrap();
    let ivalue = IValue::Tuple(vec![IValue::Tensor(features), IValue::Tensor(seq_lengths)]);
    let out = model.forward_is(&[ivalue]).unwrap();
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
        for tok in toks {
            println!("{}", tok);
        }
    }
}
