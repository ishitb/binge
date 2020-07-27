import React, { useState, useEffect } from "react";
import "../assets/main_page.css";
import Card from "../assets/card";

export default function MainPage() {
    const [headMovieTitle, setHeadMovieTitle] = useState(["Money Heist"]);
    const [headIMDB, setHeadIMDB] = useState(8.9);
    const [movieList, setMovieList] = useState([]);
    const [shows, setShows] = useState([]);

    const getShows = async () => {
        const response = await fetch(
            "https://www.episodate.com/api/most-popular?page=1"
        );
        const data = await response.json();
        
        await setShows(data.tv_shows)
    };

    useEffect(() => {
        getShows();
    }, []);

    return (
        <div>
            <header>
                <h1 className="head-title">{headMovieTitle}</h1>
                <p className="head-rating">
                    {" "}
                    <span className="rating-heading">IMDB Rating:</span>
                    {headIMDB}
                </p>
            </header>
            <section>
                {shows.map((item) => (
                    <Card
                        name={item['name']}
                        // image="https://sm.mashable.com/t/mashable_pk/photo/default/vw05sgzdlorqja6lmz6b_hcvz.960.jpg"
                        image={item['image_thumbnail_path']}
                        network={item['network']}
                    />
                ))}
            </section>
        </div>
    );
}
