import React, { useState, useEffect } from "react";
import "./card.css";

function Card({ name, image, network }) {
    return (
        <div
            className="card"
            style={
                {
                 background: `url(${image}) no-repeat center center/cover`
                }
            }
        >
            <h1>{name}</h1>
            <p>Network : {network} </p>
        </div>
    );
}

export default Card;
