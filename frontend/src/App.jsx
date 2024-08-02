import React, { useState, useEffect, useCallback } from "react";

import ImageList from "@mui/material/ImageList";
import ImageListItem from "@mui/material/ImageListItem";
import TextField from "@mui/material/TextField";
import FormControl from "@mui/material/FormControl";

import InfiniteScroll from 'react-infinite-scroll-component';

import axios from 'axios';

const Cell = ({img}) => {
  const [label, setLabel] = useState("unk");

  useEffect(() => {
    setLabel(label => "unk");
  }, [img.id]);
  
  const onClick = function () {
    console.log(img);
    if (label == "unk") {
      setLabel(label => "pos");
    } else if (label == "pos") {
      setLabel(label => "neg");
    } else if (label == "neg") {
      setLabel(label => "unk");
    }
  };

  let style;
  switch (label) {
    case "unk":
      style = {};
      break;
    case "pos":
      style = {
        border: "3px",
        borderStyle: "solid",
        borderColor: "rgba(0, 255, 0)"
      };
      break;
    case "neg":
      style = {
        border: "3px",
        borderStyle: "solid",
        borderColor: "rgba(255, 0, 0)"
      };
      break;
    default:
      style = {};
  }

  return (
    <ImageListItem style={style} key={img.id} onClick={onClick}>
      <img
        src={`${img.url}?w=164&h=164&fit=clamp&auto=format`}
        alt={img.id}
        loading="lazy"
        onError={(e) => {
          e.target.onerror                  = null;
          e.target.style.display            = "none";
          e.target.parentNode.style.display = "none";
        }}
      />
    </ImageListItem>
  );
};

const Search = function ({setImgs}) {
  const onKeyDown = (e) => {
    if (e.keyCode === 13) { // enter
      e.preventDefault();
      setImgs([]);
      
      const query = e.target.value;
      console.log('query=', query);
      
      axios.post("http://0.0.0.0:1234/knn-service", 
        {text : query, n_imgs : 1000},
        {headers : {"Content-Type" : "application/json"}}
      )
      .then((x) => {
        setImgs(x.data);
      });
    }
  };
  
  return <TextField variant="standard" label="" placeholder="" onKeyDown={onKeyDown}/>
};

function App() {
  const [imgs, setImgs] = useState([]);
  const [n_imgs, setN] = useState(100);
  
  const fetchData = () => {
    setN(n_imgs => n_imgs + 100);
  }
  
  return (
    <div>
      <div>
        <Search setImgs={setImgs}/>
      </div>
      <div>
        <InfiniteScroll
          dataLength={n_imgs}
          next={fetchData}
          hasMore={n_imgs < imgs.length}
          loader={<p>Loading...</p>}
          endMessage={<p>No more data to load.</p>}
        >
        <ImageList cols={8}>
          {
            imgs.slice(0, n_imgs).map((img) => (
              <Cell img={img}/>
            ))
          }
        </ImageList>
    </InfiniteScroll>
      </div>
    </div>
  );
}

export default App;